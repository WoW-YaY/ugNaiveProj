function varargout = Experiment7(varargin)
% EXPERIMENT7 MATLAB code for Experiment7.fig
%      EXPERIMENT7, by itself, creates a new EXPERIMENT7 or raises the existing
%      singleton*.
%
%      H = EXPERIMENT7 returns the handle to a new EXPERIMENT7 or the handle to
%      the existing singleton*.
%
%      EXPERIMENT7('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in EXPERIMENT7.M with the given input arguments.
%
%      EXPERIMENT7('Property','Value',...) creates a new EXPERIMENT7 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Experiment7_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Experiment7_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Experiment7

% Last Modified by GUIDE v2.5 11-Apr-2021 17:56:18

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Experiment7_OpeningFcn, ...
                   'gui_OutputFcn',  @Experiment7_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Experiment7 is made visible.
function Experiment7_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Experiment7 (see VARARGIN)

% Choose default command line output for Experiment7
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Experiment7 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Experiment7_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --------------------------------------------------------------------
function Experiment1_Callback(hObject, eventdata, handles)
% hObject    handle to Experiment1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=gcf; Experiment1; close(h);

% --------------------------------------------------------------------
function Experiment2_Callback(hObject, eventdata, handles)
% hObject    handle to Experiment2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=gcf; Experiment2; close(h);


% --------------------------------------------------------------------
function Experiment3_Callback(hObject, eventdata, handles)
% hObject    handle to Experiment3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=gcf; Experiment3; close(h);


% --------------------------------------------------------------------
function Experiment4_Callback(hObject, eventdata, handles)
% hObject    handle to Experiment4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=gcf; Experiment4; close(h);


% --------------------------------------------------------------------
function Experiment5_Callback(hObject, eventdata, handles)
% hObject    handle to Experiment5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=gcf; Experiment5; close(h);


% --------------------------------------------------------------------
function Experiment6_Callback(hObject, eventdata, handles)
% hObject    handle to Experiment6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=gcf; Experiment6; close(h);


% --------------------------------------------------------------------
function Experiment7_Callback(hObject, eventdata, handles)
% hObject    handle to Experiment7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=gcf; Experiment7;


% --------------------------------------------------------------------
function uipushtool1_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to uipushtool1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename,pathname]=uigetfile({'*.bmp;*.jpg;*.png;*.jpeg;*.tif'},'请选择图片');
str=[pathname filename];
OriginalPic = imread(str);
axes(handles.axes1);

imshow(OriginalPic);
set(handles.edit1,'String',str);


% --------------------------------------------------------------------
function uipushtool2_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to uipushtool2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=getimage(handles.axes3);
imwrite(h,'output_7.bmp','bmp');


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
nAmount=101;        % 滤波半径
sigma=3;            % 高斯滤波方差
threshold=0.04;     % 低对比度mask阈值
amount=20;          % 文字增强的参数

I=double(getimage(handles.axes2));
w1=fspecial('average',[nAmount,nAmount]);           % 均值滤波
avg=imfilter(I,w1);
src=I./avg;                                         % 利用均值滤波消除灰色区域
w2=fspecial('gaussian',[nAmount,nAmount],sigma);    % 高斯滤波
imgBlurred=imfilter(src,w2);
lowContrastMask = abs(src-imgBlurred)<threshold;    % 按阈值筛选低对比度区域
dst = src*(1+amount)+imgBlurred*(-amount);          % 减去高斯滤波结果
dst = double(src.*lowContrastMask+dst.*(~lowContrastMask)); % copyTo函数的效果
res = im2uint8(dst);
axes(handles.axes3);
imshow(lowContrastMask);

function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function axes1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes1
set(gca,'XColor',get(gca,'Color')) ;% 将坐标轴和坐标刻度转为白色
set(gca,'YColor',get(gca,'Color'));

set(gca,'XTickLabel',[]); % 去除坐标刻度
set(gca,'YTickLabel',[]);


% --- Executes during object creation, after setting all properties.
function axes2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes2
set(gca,'XColor',get(gca,'Color')) ;% 将坐标轴和坐标刻度转为白色
set(gca,'YColor',get(gca,'Color'));

set(gca,'XTickLabel',[]); % 去除坐标刻度
set(gca,'YTickLabel',[]);


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
a=get(handles.axes1,'position');
OriginalPic=getimage(handles.axes1);
NewPic=imresize(OriginalPic,(a(3)-0.02)/a(3),'bicubic');
imshow(NewPic);
% set(handles.axes1,'units','normalized','position',[a(1)+0.01 a(2)+0.01 a(3)-0.02 a(4)-0.02]);

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
a=get(handles.axes1,'position');
OriginalPic=getimage(handles.axes1);
NewPic=imresize(OriginalPic,(a(3)+0.02)/a(3),'bicubic');
imshow(NewPic);
% set(handles.axes1,'units','normalized','position',[a(1)-0.01 a(2)-0.01 a(3)+0.02 a(4)+0.02]);

% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
A = getimage(handles.axes1);
data = get(handles.slider1,'value');
B = imrotate(A,data);
axes(handles.axes2);
imshow(B);


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --------------------------------------------------------------------
function uitoggletool1_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to uitoggletool1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in togglebutton1.
function togglebutton1_Callback(hObject, eventdata, handles)
% hObject    handle to togglebutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of togglebutton1


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
A = getimage(handles.axes2);
s = size(A);
if s(3) == 3
    A = rgb2gray(A);
end
axes(handles.axes2);
imshow(A);


% --- Executes on mouse press over figure background, over a disabled or
% --- inactive control, or over an axes background.
function figure1_WindowButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
A = getimage(handles.axes2);
edit = ginput();
rect = [edit(1,1) edit(1,2) edit(2,1) edit(2,2)];
B=imcrop(A,rect);
rectangle('Position',rect,'LineWidth',2,'EdgeColor','r');
axes(handles.axes2);
imshow(B);


function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
img = imread(file);
axes(handles.axes1);
imshow(img);


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.edit4,'string',num2str([]));


% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
img=getimage(handles.axes2);
[H W]=size(img);
p1=ginput();        %依次点击左上、右上、左下、右下
p2=[1,1;W,1;1,H;W,H];

T=calc_homography(p1,p2);   %计算单应性矩阵
T=maketform('projective',T);   %投影矩阵
[imgn X Y]=imtransform(img,T);     %投影

axes(handles.axes2);
imshow(imgn,[]);

% --- Executes on mouse motion over figure - except title and menu.
function figure1_WindowButtonMotionFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
