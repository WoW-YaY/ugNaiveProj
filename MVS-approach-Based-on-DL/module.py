import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastMVSNet(nn.Module):
    def __init__(self,
                 img_base_channels=8,
                 vol_base_channels=8,
                 flow_channels=(64, 64, 16, 1),
                 k=16,
                 ):
        super(FastMVSNet, self).__init__()
        self.k = k

        self.feature_fetcher = FeatureFetcher()
        self.feature_grad_fetcher = FeatureGradFetcher()
        self.point_grad_fetcher = PointGrad()

        self.coarse_img_conv = ImageConv(img_base_channels)
        self.coarse_vol_conv = VolumeConv(img_base_channels * 4, vol_base_channels)
        self.propagation_net = PropagationNet(img_base_channels)
        self.flow_img_conv = ImageConv(img_base_channels)

    def forward(self, data_batch, img_scales, inter_scales, isGN, isTest=False):
        preds = collections.OrderedDict()
        img_list = data_batch["img_list"]
        cam_params_list = data_batch["cam_params_list"]

        cam_extrinsic = cam_params_list[:, :, 0, :3, :4].clone()  # (B, V, 3, 4)
        R = cam_extrinsic[:, :, :3, :3]
        t = cam_extrinsic[:, :, :3, 3].unsqueeze(-1)
        R_inv = torch.inverse(R)
        cam_intrinsic = cam_params_list[:, :, 1, :3, :3].clone()

        if isTest:
            cam_intrinsic[:, :, :2, :3] = cam_intrinsic[:, :, :2, :3] / 4.0

        depth_start = cam_params_list[:, 0, 1, 3, 0]
        depth_interval = cam_params_list[:, 0, 1, 3, 1]
        num_depth = cam_params_list[0, 0, 1, 3, 2].long()

        depth_end = depth_start + (num_depth - 1) * depth_interval

        batch_size, num_view, img_channel, img_height, img_width = list(img_list.size())

        coarse_feature_maps = []
        for i in range(num_view):
            curr_img = img_list[:, i, :, :, :]
            curr_feature_map = self.coarse_img_conv(curr_img)["conv2"]
            coarse_feature_maps.append(curr_feature_map)

        feature_list = torch.stack(coarse_feature_maps, dim=1)

        feature_channels, feature_height, feature_width = list(curr_feature_map.size())[1:]

        depths = []
        for i in range(batch_size):
            depths.append(torch.linspace(depth_start[i], depth_end[i], num_depth, device=img_list.device) \
                          .view(1, 1, num_depth, 1))
        depths = torch.stack(depths, dim=0)  # (B, 1, 1, D, 1)

        feature_map_indices_grid = get_pixel_grids(feature_height, feature_width)
        # print("before:", feature_map_indices_grid.size())
        feature_map_indices_grid = feature_map_indices_grid.view(1, 3, feature_height, feature_width)[:, :, ::2, ::2].contiguous()
        # print("after:", feature_map_indices_grid.size())
        feature_map_indices_grid = feature_map_indices_grid.view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(img_list.device)

        ref_cam_intrinsic = cam_intrinsic[:, 0, :, :].clone()
        uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1), feature_map_indices_grid)  # (B, 1, 3, FH*FW)

        cam_points = (uv.unsqueeze(3) * depths).view(batch_size, 1, 3, -1)  # (B, 1, 3, D*FH*FW)
        world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2).contiguous() \
            .view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

        preds["world_points"] = world_points

        num_world_points = world_points.size(-1)
        assert num_world_points == feature_height * feature_width * num_depth / 4

        point_features = self.feature_fetcher(feature_list, world_points, cam_intrinsic, cam_extrinsic)
        ref_feature = coarse_feature_maps[0]
        #print("before ref feature:", ref_feature.size())
        ref_feature = ref_feature[:, :, ::2,::2].contiguous()
        #print("after ref feature:", ref_feature.size())
        ref_feature = ref_feature.unsqueeze(2).expand(-1, -1, num_depth, -1, -1)\
                        .contiguous().view(batch_size,feature_channels,-1)
        point_features[:, 0, :, :] = ref_feature

        avg_point_features = torch.mean(point_features, dim=1)
        avg_point_features_2 = torch.mean(point_features ** 2, dim=1)

        point_features = avg_point_features_2 - (avg_point_features ** 2)

        cost_volume = point_features.view(batch_size, feature_channels, num_depth, feature_height // 2, feature_width // 2)

        filtered_cost_volume = self.coarse_vol_conv(cost_volume).squeeze(1)

        probability_volume = F.softmax(-filtered_cost_volume, dim=1)
        depth_volume = []
        for i in range(batch_size):
            depth_array = torch.linspace(depth_start[i], depth_end[i], num_depth, device=depth_start.device)
            depth_volume.append(depth_array)
        depth_volume = torch.stack(depth_volume, dim=0)  # (B, D)
        depth_volume = depth_volume.view(batch_size, num_depth, 1, 1).expand(probability_volume.shape)
        pred_depth_img = torch.sum(depth_volume * probability_volume, dim=1).unsqueeze(1)  # (B, 1, FH, FW)

        prob_map = get_propability_map(probability_volume, pred_depth_img, depth_start, depth_interval)

        # image guided depth map propagation
        pred_depth_img = F.interpolate(pred_depth_img, (feature_height, feature_width), mode="nearest")
        prob_map = F.interpolate(prob_map, (feature_height, feature_width), mode="bilinear")
        pred_depth_img = self.propagation_net(pred_depth_img, img_list[:, 0, :, :, :])

        preds["coarse_depth_map"] = pred_depth_img
        preds["coarse_prob_map"] = prob_map

        if isGN:
            feature_pyramids = {}
            chosen_conv = ["conv1", "conv2"]
            for conv in chosen_conv:
                feature_pyramids[conv] = []
            for i in range(num_view):
                curr_img = img_list[:, i, :, :, :]
                curr_feature_pyramid = self.flow_img_conv(curr_img)
                for conv in chosen_conv:
                    feature_pyramids[conv].append(curr_feature_pyramid[conv])

            for conv in chosen_conv:
                feature_pyramids[conv] = torch.stack(feature_pyramids[conv], dim=1)

            if isTest:
                for conv in chosen_conv:
                    feature_pyramids[conv] = torch.detach(feature_pyramids[conv])


            def gn_update(estimated_depth_map, interval, image_scale, it):
                nonlocal chosen_conv
                # print(estimated_depth_map.size(), image_scale)
                flow_height, flow_width = list(estimated_depth_map.size())[2:]
                if flow_height != int(img_height * image_scale):
                    flow_height = int(img_height * image_scale)
                    flow_width = int(img_width * image_scale)
                    estimated_depth_map = F.interpolate(estimated_depth_map, (flow_height, flow_width), mode="nearest")
                else:
                    # if it is the same size return directly
                    return estimated_depth_map
                    # pass
                
                if isTest:
                    estimated_depth_map = estimated_depth_map.detach()

                # GN step
                cam_intrinsic = cam_params_list[:, :, 1, :3, :3].clone()
                if isTest:
                    cam_intrinsic[:, :, :2, :3] *= image_scale
                else:
                    cam_intrinsic[:, :, :2, :3] *= (4 * image_scale)

                ref_cam_intrinsic = cam_intrinsic[:, 0, :, :].clone()
                feature_map_indices_grid = get_pixel_grids(flow_height, flow_width) \
                    .view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(img_list.device)

                uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1),
                                  feature_map_indices_grid)  # (B, 1, 3, FH*FW)

                interval_depth_map = estimated_depth_map
                cam_points = (uv * interval_depth_map.view(batch_size, 1, 1, -1))
                world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2) \
                    .contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

                grad_pts = self.point_grad_fetcher(world_points, cam_intrinsic, cam_extrinsic)

                R_tar_ref = torch.bmm(R.view(batch_size * num_view, 3, 3),
                                      R_inv[:, 0:1, :, :].repeat(1, num_view, 1, 1).view(batch_size * num_view, 3, 3))

                R_tar_ref = R_tar_ref.view(batch_size, num_view, 3, 3)
                d_pts_d_d = uv.unsqueeze(-1).permute(0, 1, 3, 2, 4).contiguous().repeat(1, num_view, 1, 1, 1)
                d_pts_d_d = R_tar_ref.unsqueeze(2) @ d_pts_d_d
                d_uv_d_d = torch.bmm(grad_pts.view(-1, 2, 3), d_pts_d_d.view(-1, 3, 1)).view(batch_size, num_view, 1,
                                                                                             -1, 2, 1)
                all_features = []
                for conv in chosen_conv:
                    curr_feature = feature_pyramids[conv]
                    c, h, w = list(curr_feature.size())[2:]
                    curr_feature = curr_feature.contiguous().view(-1, c, h, w)
                    curr_feature = F.interpolate(curr_feature, (flow_height, flow_width), mode="bilinear")
                    curr_feature = curr_feature.contiguous().view(batch_size, num_view, c, flow_height, flow_width)

                    all_features.append(curr_feature)

                all_features = torch.cat(all_features, dim=2)

                if isTest:
                    point_features, point_features_grad = \
                        self.feature_grad_fetcher.test_forward(all_features, world_points, cam_intrinsic, cam_extrinsic)
                else:
                    point_features, point_features_grad = \
                        self.feature_grad_fetcher(all_features, world_points, cam_intrinsic, cam_extrinsic)

                c = all_features.size(2)
                d_uv_d_d_tmp = d_uv_d_d.repeat(1, 1, c, 1, 1, 1)
                # print("d_uv_d_d tmp size:", d_uv_d_d_tmp.size())
                J = point_features_grad.view(-1, 1, 2) @ d_uv_d_d_tmp.view(-1, 2, 1)
                J = J.view(batch_size, num_view, c, -1, 1)[:, 1:, ...].contiguous()\
                    .permute(0, 3, 1, 2, 4).contiguous().view(-1, c * (num_view - 1), 1)

                # print(J.size())
                resid = point_features[:, 1:, ...] - point_features[:, 0:1, ...]
                first_resid = torch.sum(torch.abs(resid), dim=(1, 2))
                # print(resid.size())
                resid = resid.permute(0, 3, 1, 2).contiguous().view(-1, c * (num_view - 1), 1)

                J_t = torch.transpose(J, 1, 2)
                H = J_t @ J
                b = -J_t @ resid
                delta = b / (H + 1e-6)
                # #print(delta.size())
                _, _, h, w = estimated_depth_map.size()
                flow_result = estimated_depth_map  + delta.view(-1, 1, h, w)

                # check update results
                interval_depth_map = flow_result
                cam_points = (uv * interval_depth_map.view(batch_size, 1, 1, -1))
                world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2) \
                    .contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

                point_features = \
                    self.feature_fetcher(all_features, world_points, cam_intrinsic, cam_extrinsic)

                resid = point_features[:, 1:, ...] - point_features[:, 0:1, ...]
                second_resid = torch.sum(torch.abs(resid), dim=(1, 2))
                # print(first_resid.size(), second_resid.size())

                # only accept good update
                flow_result = torch.where((second_resid < first_resid).view(batch_size, 1, flow_height, flow_width),
                                          flow_result, estimated_depth_map)
                return flow_result

            for i, (img_scale, inter_scale) in enumerate(zip(img_scales, inter_scales)):
                if isTest:
                    pred_depth_img = torch.detach(pred_depth_img)
                    print("update: {}".format(i))
                flow = gn_update(pred_depth_img, inter_scale* depth_interval, img_scale, i)
                preds["flow{}".format(i+1)] = flow
                pred_depth_img = flow

        return preds


class PointMVSNetLoss(nn.Module):
    def __init__(self, valid_threshold):
        super(PointMVSNetLoss, self).__init__()
        self.maeloss = MAELoss()
        self.valid_maeloss = Valid_MAELoss(valid_threshold)

    def forward(self, preds, labels, isFlow):
        gt_depth_img = labels["gt_depth_img"]
        depth_interval = labels["cam_params_list"][:, 0, 1, 3, 1]

        coarse_depth_map = preds["coarse_depth_map"]
        resize_gt_depth = F.interpolate(gt_depth_img, (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))
        coarse_loss = self.maeloss(coarse_depth_map, resize_gt_depth, depth_interval)

        losses = {}
        losses["coarse_loss"] = coarse_loss

        if isFlow:
            flow1 = preds["flow1"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow1.shape[2], flow1.shape[3]))
            flow1_loss = self.maeloss(flow1, resize_gt_depth, 0.75 * depth_interval)
            losses["flow1_loss"] = flow1_loss

            flow2 = preds["flow2"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow2.shape[2], flow2.shape[3]))
            flow2_loss = self.maeloss(flow2, resize_gt_depth, 0.375 * depth_interval)
            losses["flow2_loss"] = flow2_loss

        for k in losses.keys():
            losses[k] /= float(len(losses.keys()))

        return losses


def cal_less_percentage(pred_depth, gt_depth, depth_interval, threshold):
    shape = list(pred_depth.size())
    mask_valid = (~torch.eq(gt_depth, 0.0)).type(torch.float)
    denom = torch.sum(mask_valid) + 1e-7
    interval_image = depth_interval.view(-1, 1, 1, 1).expand(shape)
    abs_diff_image = torch.abs(pred_depth - gt_depth) / interval_image

    pct = mask_valid * (abs_diff_image <= threshold).type(torch.float)

    pct = torch.sum(pct) / denom

    return pct


def cal_valid_less_percentage(pred_depth, gt_depth, before_depth, depth_interval, threshold, valid_threshold):
    shape = list(pred_depth.size())
    mask_true = (~torch.eq(gt_depth, 0.0)).type(torch.float)
    interval_image = depth_interval.view(-1, 1, 1, 1).expand(shape)
    abs_diff_image = torch.abs(pred_depth - gt_depth) / interval_image

    if before_depth.size(2) != shape[2]:
        before_depth = F.interpolate(before_depth, (shape[2], shape[3]))

    diff = torch.abs(before_depth - gt_depth) / interval_image
    mask_valid = (diff < valid_threshold).type(torch.float)
    mask_valid = mask_valid * mask_true

    denom = torch.sum(mask_valid) + 1e-7
    pct = mask_valid * (abs_diff_image <= threshold).type(torch.float)

    pct = torch.sum(pct) / denom

    return pct


class PointMVSNetMetric(nn.Module):
    def __init__(self, valid_threshold):
        super(PointMVSNetMetric, self).__init__()
        self.valid_threshold = valid_threshold

    def forward(self, preds, labels, isFlow):
        gt_depth_img = labels["gt_depth_img"]
        depth_interval = labels["cam_params_list"][:, 0, 1, 3, 1]

        coarse_depth_map = preds["coarse_depth_map"]
        resize_gt_depth = F.interpolate(gt_depth_img, (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))

        less_one_pct_coarse = cal_less_percentage(coarse_depth_map, resize_gt_depth, depth_interval, 1.0)
        less_three_pct_coarse = cal_less_percentage(coarse_depth_map, resize_gt_depth, depth_interval, 3.0)

        metrics = {
            "<1_pct_cor": less_one_pct_coarse,
            "<3_pct_cor": less_three_pct_coarse,
        }

        if isFlow:
            flow1 = preds["flow1"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow1.shape[2], flow1.shape[3]))

            less_one_pct_flow1 = cal_valid_less_percentage(flow1, resize_gt_depth, coarse_depth_map,
                                                           0.75 * depth_interval, 1.0, self.valid_threshold)
            less_three_pct_flow1 = cal_valid_less_percentage(flow1, resize_gt_depth, coarse_depth_map,
                                                             0.75 * depth_interval, 3.0, self.valid_threshold)

            metrics["<1_pct_flow1"] = less_one_pct_flow1
            metrics["<3_pct_flow1"] = less_three_pct_flow1

            flow2 = preds["flow2"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow2.shape[2], flow2.shape[3]))

            less_one_pct_flow2 = cal_valid_less_percentage(flow2, resize_gt_depth, flow1,
                                                           0.375 * depth_interval, 1.0, self.valid_threshold)
            less_three_pct_flow2 = cal_valid_less_percentage(flow2, resize_gt_depth, flow1,
                                                             0.375 * depth_interval, 3.0, self.valid_threshold)

            metrics["<1_pct_flow2"] = less_one_pct_flow2
            metrics["<3_pct_flow2"] = less_three_pct_flow2

        return metrics


def build_pointmvsnet(cfg):
    net = FastMVSNet(
        img_base_channels=cfg.MODEL.IMG_BASE_CHANNELS,
        vol_base_channels=cfg.MODEL.VOL_BASE_CHANNELS,
        flow_channels=cfg.MODEL.FLOW_CHANNELS,
    )

    loss_fn = PointMVSNetLoss(
        valid_threshold=cfg.MODEL.VALID_THRESHOLD,
    )

    metric_fn = PointMVSNetMetric(
        valid_threshold=cfg.MODEL.VALID_THRESHOLD,
    )

    return net, loss_fn, metric_fn



def mkdir(path):
    os.makedirs(path, exist_ok=True)


def load_cam_dtu(file, num_depth=0, interval_scale=1.0):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    words = file.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = num_depth
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (num_depth - 1)
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * (num_depth - 1)
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam


def write_cam_dtu(file, cam):
    # f = open(file, "w")
    f = open(file, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write(
        '\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()


def load_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def write_pfm(file, image, scale=1):
    file = open(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)

    file.close()


def setup_logger(name, save_dir, prefix="", timestamp=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        timestamp = time.strftime(".%m_%d_%H_%M_%S") if timestamp else ""
        prefix = "." + prefix if prefix else ""
        log_file = os.path.join(save_dir, "log{}.txt".format(prefix + timestamp))
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def shutdown_logger(logger):
    logger.handlers = []

class AverageMeter(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.values = deque(maxlen=window_size)
        self.counts = deque(maxlen=window_size)
        self.sum = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.values.append(value)
        self.counts.append(count)
        self.sum += value
        self.count += count

    @property
    def avg(self):
        if np.sum(self.counts) == 0:
            return 0
        return np.sum(self.values) / np.sum(self.counts)

    @property
    def global_avg(self):
        if self.count == 0:
            return 0
        return self.sum / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            count = 1
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    count = v.numel()
                    v = v.sum().item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, count)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)

    def __str__(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.avg, meter.global_avg)
            )
        return self.delimiter.join(metric_str)

    @property
    def summary_str(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(metric_str)
    
def eval_file_logger(data_batch, preds, ref_img_path, folder, scene_name_index=-2, out_index_minus=1, save_prob_volume=False):
    l = ref_img_path.split("/")
    eval_folder = "/".join(l[:-3])

    scene = l[scene_name_index]

    scene_folder = osp.join(eval_folder, folder, scene)

    if not osp.isdir(scene_folder):
        mkdir(scene_folder)
        print("**** {} ****".format(scene))

    out_index = int(l[-1][5:8]) - out_index_minus

    cam_params_list = data_batch["cam_params_list"].cpu().numpy()

    ref_cam_paras = cam_params_list[0, 0, :, :, :]

    init_depth_map_path = scene_folder + ('/%08d_init.pfm' % out_index)
    init_prob_map_path = scene_folder + ('/%08d_init_prob.pfm' % out_index)
    out_ref_image_path = scene_folder + ('/%08d.jpg' % out_index)

    init_depth_map = preds["coarse_depth_map"].cpu().numpy()[0, 0]
    init_prob_map = preds["coarse_prob_map"].cpu().numpy()[0, 0]
    ref_image = data_batch["ref_img"][0].cpu().numpy()

    write_pfm(init_depth_map_path, init_depth_map)
    write_pfm(init_prob_map_path, init_prob_map)
    cv2.imwrite(out_ref_image_path, ref_image)

    out_init_cam_path = scene_folder + ('/cam_%08d_init.txt' % out_index)
    init_cam_paras = ref_cam_paras.copy()
    init_cam_paras[1, :2, :3] *= (float(init_depth_map.shape[0]) / ref_image.shape[0])
    write_cam_dtu(out_init_cam_path, init_cam_paras)

    interval_list = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    interval_list = np.reshape(interval_list, [1, 1, -1])

    for i, k in enumerate(preds.keys()):
        if "flow" in k:
            if "prob" in k:
                out_flow_prob_map = preds[k][0].cpu().permute(1, 2, 0).numpy()
                num_interval = out_flow_prob_map.shape[-1]
                assert num_interval == interval_list.size
                pred_interval = np.sum(out_flow_prob_map * interval_list, axis=-1) + 2.0
                pred_floor = np.floor(pred_interval).astype(np.int)[..., np.newaxis]
                pred_ceil = pred_floor + 1
                pred_ceil = np.clip(pred_ceil, 0, num_interval - 1)
                pred_floor = np.clip(pred_floor, 0, num_interval - 1)
                prob_height, prob_width = pred_floor.shape[:2]
                prob_height_ind = np.tile(np.reshape(np.arange(prob_height), [-1, 1, 1]), [1, prob_width, 1])
                prob_width_ind = np.tile(np.reshape(np.arange(prob_width), [1, -1, 1]), [prob_height, 1, 1])

                floor_prob = np.squeeze(out_flow_prob_map[prob_height_ind, prob_width_ind, pred_floor], -1)
                ceil_prob = np.squeeze(out_flow_prob_map[prob_height_ind, prob_width_ind, pred_ceil], -1)
                flow_prob = floor_prob + ceil_prob
                flow_prob_map_path = scene_folder + "/{:08d}_{}.pfm".format(out_index, k)
                write_pfm(flow_prob_map_path, flow_prob)

            else:
                out_flow_depth_map = preds[k][0, 0].cpu().numpy()
                flow_depth_map_path = scene_folder + "/{:08d}_{}.pfm".format(out_index, k)
                write_pfm(flow_depth_map_path, out_flow_depth_map)
                out_flow_cam_path = scene_folder + "/cam_{:08d}_{}.txt".format(out_index, k)
                flow_cam_paras = ref_cam_paras.copy()
                flow_cam_paras[1, :2, :3] *= (float(out_flow_depth_map.shape[0]) / float(ref_image.shape[0]))
                write_cam_dtu(out_flow_cam_path, flow_cam_paras)

                world_pts = depth2pts_np(out_flow_depth_map, flow_cam_paras[1][:3, :3], flow_cam_paras[0])
                save_points(osp.join(scene_folder, "{:08d}_{}pts.xyz".format(out_index, k)), world_pts)
    # save cost volume
    if save_prob_volume:
        probability_volume = preds["coarse_prob_volume"].cpu().numpy()[0]
        init_prob_volume_path = scene_folder + ('/%08d_init_prob_volume.npz' % out_index)
        np.savez(init_prob_volume_path, probability_volume)


def depth2pts_np(depth_map, cam_intrinsic, cam_extrinsic):
    feature_grid = get_pixel_grids_np(depth_map.shape[0], depth_map.shape[1])

    uv = np.matmul(np.linalg.inv(cam_intrinsic), feature_grid)
    cam_points = uv * np.reshape(depth_map, (1, -1))

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3:4]
    R_inv = np.linalg.inv(R)

    world_points = np.matmul(R_inv, cam_points - t).transpose()
    return world_points


def get_pixel_grids_np(height, width):
    x_linspace = np.linspace(0.5, width - 0.5, width)
    y_linspace = np.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
    x_coordinates = np.reshape(x_coordinates, (1, -1))
    y_coordinates = np.reshape(y_coordinates, (1, -1))
    ones = np.ones_like(x_coordinates).astype(np.float)
    grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)

    return grid


def save_points(path, points):
    np.savetxt(path, points, delimiter=' ', fmt='%.4f')
