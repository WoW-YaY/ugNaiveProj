function varargout = Experiment5(varargin)
% EXPERIMENT5 MATLAB code for Experiment5.fig
%      EXPERIMENT5, by itself, creates a new EXPERIMENT5 or raises the existing
%      singleton*.
%
%      H = EXPERIMENT5 returns the handle to a new EXPERIMENT5 or the handle to
%      the existing singleton*.
%
%      EXPERIMENT5('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in EXPERIMENT5.M with the given input arguments.
%
%      EXPERIMENT5('Property','Value',...) creates a new EXPERIMENT5 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Experiment5_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Experiment5_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Experiment5

% Last Modified by GUIDE v2.5 09-Apr-2021 12:34:46

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Experiment5_OpeningFcn, ...
                   'gui_OutputFcn',  @Experiment5_OutputFcn, ...
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


% --- Executes just before Experiment5 is made visible.
function Experiment5_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Experiment5 (see VARARGIN)

% Choose default command line output for Experiment5
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Experiment5 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Experiment5_OutputFcn(hObject, eventdata, handles) 
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
h=gcf; Experiment5;



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
h=getimage(handles.axes6);
imwrite(h,'output_5.bmp','bmp');


% --- Executes during object creation, after setting all properties.
function axes1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(gca,'XColor',get(gca,'Color')) ;% 将坐标轴和坐标刻度转为白色
set(gca,'YColor',get(gca,'Color'));

set(gca,'XTickLabel',[]); % 去除坐标刻度
set(gca,'YTickLabel',[]);
% Hint: place code in OpeningFcn to populate axes1


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
img = imread(get(handles.edit1,'string'));
h = imhist(img)/numel(img);
h1 = h(1:1:256);
horz=1:1:256;
axes(handles.axes2);
bar(horz,h);


% --- Executes during object creation, after setting all properties.
function axes2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(gca,'XColor',get(gca,'Color')) ;% 将坐标轴和坐标刻度转为白色
set(gca,'YColor',get(gca,'Color'));

set(gca,'XTickLabel',[]); % 去除坐标刻度
set(gca,'YTickLabel',[]);
% Hint: place code in OpeningFcn to populate axes2


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
T = str2num(get(handles.edit2,'string'))/100;
file = get(handles.edit1,'string');
f = imread(file)/256;
[gx gy] = imgradientxy(f, 'sobel');
M = (gx.^2+gy.^2).^0.5;
num = int64(numel(M) * T);
sort_M = sort(M(:));
threshold = sort_M(num);
C = zeros(size(f));
[H,W] = size(f);
for i = 1:H
    for j = 1:W
        if M(i,j)>threshold
            C(i,j)=1;
        end
    end
end
axes(handles.axes3);
imshow(C);


% --- Executes during object creation, after setting all properties.
function axes3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(gca,'XColor',get(gca,'Color')) ;% 将坐标轴和坐标刻度转为白色
set(gca,'YColor',get(gca,'Color'));

set(gca,'XTickLabel',[]); % 去除坐标刻度
set(gca,'YTickLabel',[]);
% Hint: place code in OpeningFcn to populate axes3


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
f = imread(file);
img_C = getimage(handles.axes3);
img_D = f .* uint8(img_C);
axes(handles.axes4);
imshow(img_D);

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
img_D = getimage(handles.axes4);
h = imhist(img_D) / numel(img_D);
h(1) = 0;
h = h / sum(h);
h1 = h(1:1:256);
horz=1:1:256;
axes(handles.axes5);
bar(horz,h);


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
f = imread(file);
img_D = getimage(handles.axes4);
h = imhist(img_D) / numel(img_D);
h(1) = 0;
h = h / sum(h);
h1 = h(1:1:256);
P1(1) = h1(1);
m(1) = h1(1);
for k = 2:length(h1)
    P1(k) = P1(k-1) + h1(k);
    m(k) = m(k-1) + k*h1(k);
end
mG = m(length(h1));
for k = 1:length(h1)
    sigmaB2(k) = (mG*P1(k)-m(k))^2/(P1(k)*(1-P1(k)));
end
maximum = max(sigmaB2); 
K = find(sigmaB2==maximum);
k_star = mean(K);
set(handles.edit3,'string',num2str(uint8(k_star)));
axes(handles.axes6);
final = im2bw(f,k_star/256);
imshow(final);


function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function axes4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(gca,'XColor',get(gca,'Color')) ;% 将坐标轴和坐标刻度转为白色
set(gca,'YColor',get(gca,'Color'));

set(gca,'XTickLabel',[]); % 去除坐标刻度
set(gca,'YTickLabel',[]);
% Hint: place code in OpeningFcn to populate axes4


% --- Executes during object creation, after setting all properties.
function axes5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(gca,'XColor',get(gca,'Color')) ;% 将坐标轴和坐标刻度转为白色
set(gca,'YColor',get(gca,'Color'));

set(gca,'XTickLabel',[]); % 去除坐标刻度
set(gca,'YTickLabel',[]);
% Hint: place code in OpeningFcn to populate axes5


% --- Executes during object creation, after setting all properties.
function axes6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(gca,'XColor',get(gca,'Color')) ;% 将坐标轴和坐标刻度转为白色
set(gca,'YColor',get(gca,'Color'));

set(gca,'XTickLabel',[]); % 去除坐标刻度
set(gca,'YTickLabel',[]);
% Hint: place code in OpeningFcn to populate axes6


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
f = imread(file);
h = imhist(f) / numel(f);
% h(1) = 0;
h = h / sum(h);
h1 = h(1:1:256);
P1(1) = h1(1);
m(1) = h1(1);
for k = 2:length(h1)
    P1(k) = P1(k-1) + h1(k);
    m(k) = m(k-1) + k*h1(k);
end
mG = m(length(h1));
for k = 1:length(h1)
    sigmaB2(k) = (mG*P1(k)-m(k))^2/(P1(k)*(1-P1(k)));
end
maximum = max(sigmaB2); 
K = find(sigmaB2==maximum);
k_star = mean(K);
set(handles.edit4,'string',num2str(k_star));
axes(handles.axes3);
final = im2bw(f,k_star/256);
imshow(final);



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


% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
T = str2num(get(handles.edit5,'string'))/100;
file = get(handles.edit1,'string');
f = im2double(imread(file));
w = fspecial('laplacian',0);
g = abs(imfilter(f,w,'replicate'));
num = int64(numel(g) * T);
sort_g = sort(g(:));
threshold = sort_g(num);
D = zeros(size(f));
[H,W] = size(f);
for i = 1:H
    for j = 1:W
        if g(i,j)>threshold
            D(i,j)=1;
        end
    end
end
axes(handles.axes4);
imshow(D);



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
f = imread(file);
img_D = getimage(handles.axes4);
img_E = f .* uint8(img_D);

h = imhist(img_E) / numel(img_E);
h(1) = 0;
h = h / sum(h);
h1 = h(1:1:256);
horz=1:1:256;
axes(handles.axes5);
bar(horz,h);


% --- Executes during object creation, after setting all properties.
function pushbutton6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
f = imread(file);
img_D = getimage(handles.axes4);
img_E = f .* uint8(img_D);
h = imhist(img_E) / numel(img_E);
h(1) = 0;
h = h / sum(h);
h1 = h(1:1:256);
P1(1) = h1(1);
m(1) = h1(1);
for k = 2:length(h1)
    P1(k) = P1(k-1) + h1(k);
    m(k) = m(k-1) + k*h1(k);
end
mG = m(length(h1));
for k = 1:length(h1)
    sigmaB2(k) = (mG*P1(k)-m(k))^2/(P1(k)*(1-P1(k)));
end
maximum = max(sigmaB2); 
K = find(sigmaB2==maximum);
k_star = mean(K);
set(handles.edit6,'string',num2str(k_star));
axes(handles.axes6);
final = im2bw(f,k_star/256);
imshow(final);


function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function pushbutton8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


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
h=gcf; Experiment7; close(h);
