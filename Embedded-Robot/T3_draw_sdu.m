clc;clear;close all;

% 在球面上实现"山大"的轨迹

% 设置机械臂构型
L1 = Link([0 151.9 0 pi/2]);
L2 = Link([0 0 243.7 0]);
L3 = Link([0 0 213.2 0]);
L4 = Link([0 112.4 0 pi/2]);
L5 = Link([0 85.35 0 pi/2]);
L6 = Link([0 81.9 0 0]);

L1.qlim = [-2*pi, 2*pi];
L2.qlim = [-2*pi, 2*pi];
L3.qlim = [-2*pi, 2*pi];
L4.qlim = [-2*pi, 2*pi];
L5.qlim = [-2*pi, 2*pi];
L6.qlim = [-2*pi, 2*pi];

UR3 = SerialLink([L1 L2 L3 L4 L5 L6],'name','UR3');
view(3);
hold on;

% 设定关节角初值
q0 = [90 0 0 0 0 0];
UR3.plot(q0,'tilesize',10);
hold on;

% 创建球面
[x, y, z] = sphere();
centerx = 700;
centery = 100;
centerz = 0;
r = 300;
surf(r*x+centerx, r*y+centery, r*z+centerz);
hold on;

% "山"第一划
trace1 = xlsread("tracks.xlsx","Sheet1");
plot3(trace1(1,:),trace1(2,:),trace1(3,:),'r');
T1 = transl(trace1');
I1 = UR3.ikine(T1,'mask',[1 1 1 0 0 0]);
hold on;
UR3.plot(I1,'tilesize',100);

% "山"第二划
trace2 = xlsread("tracks.xlsx","Sheet2");
plot3(trace2(1,:),trace2(2,:),trace2(3,:),'r');
T2 = transl(trace2');
I2 = UR3.ikine(T2,'mask',[1 1 1 0 0 0]);
hold on;
UR3.plot(I2,'tilesize',100);

% "大"第一划
trace3 = xlsread("tracks.xlsx","Sheet3");
plot3(trace3(1,:),trace3(2,:),trace3(3,:),'r');
T3 = transl(trace3');
I3 = UR3.ikine(T3,'mask',[1 1 1 0 0 0]);
hold on;
UR3.plot(I3,'tilesize',100);

% "大"第二划
trace4 = xlsread("tracks.xlsx","Sheet4");
plot3(trace4(1,:),trace4(2,:),trace4(3,:),'r');
T4 = transl(trace4');
I4 = UR3.ikine(T4,'mask',[1 1 1 0 0 0]);
hold on;
UR3.plot(I4,'tilesize',100);

% "大"第三划
trace5 = xlsread("tracks.xlsx","Sheet5");
plot3(trace5(1,:),trace5(2,:),trace5(3,:),'r');
T5 = transl(trace5');
I5 = UR3.ikine(T5,'mask',[1 1 1 0 0 0]);
hold on;
UR3.plot(I5,'tilesize',100);