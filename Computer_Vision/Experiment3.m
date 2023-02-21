function varargout = Experiment3(varargin)
% EXPERIMENT3 MATLAB code for Experiment3.fig
%      EXPERIMENT3, by itself, creates a new EXPERIMENT3 or raises the existing
%      singleton*.
%
%      H = EXPERIMENT3 returns the handle to a new EXPERIMENT3 or the handle to
%      the existing singleton*.
%
%      EXPERIMENT3('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in EXPERIMENT3.M with the given input arguments.
%
%      EXPERIMENT3('Property','Value',...) creates a new EXPERIMENT3 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Experiment3_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Experiment3_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Experiment3

% Last Modified by GUIDE v2.5 09-Apr-2021 12:31:46

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Experiment3_OpeningFcn, ...
                   'gui_OutputFcn',  @Experiment3_OutputFcn, ...
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


% --- Executes just before Experiment3 is made visible.
function Experiment3_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Experiment3 (see VARARGIN)

% Choose default command line output for Experiment3
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Experiment3 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Experiment3_OutputFcn(hObject, eventdata, handles) 
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
h=gcf; Experiment3;


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
function uipushtool2_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to uipushtool2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=getimage(handles.axes1);
imwrite(h,'output_3.bmp','bmp');


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
file = get(handles.edit1,'string');
f = imread(file);
[M,N] = size(f);
fillimage = uint8(zeros(2*M,2*N));
fillimage(1:M,1:N) = f;
for x = 1:2*M
    for y = 1:2*N
        h(x,y) = (-1)^(x+y);
    end
end
fillimagecenter = h.*double(fillimage);
imshow(fillimagecenter);


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
img = imread(file);
imshow(img);


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
f = imread(file);
[M,N] = size(f);
fillimage = uint8(zeros(2*M,2*N));
fillimage(1:M,1:N) = f;
imshow(fillimage);


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
f = imread(file);
[M,N] = size(f);
fillimage = uint8(zeros(2*M,2*N));
fillimage(1:M,1:N) = f;
for x = 1:2*M
    for y = 1:2*N
        h(x,y) = (-1)^(x+y);
    end
end
fillimagecenter = h.*double(fillimage);
F = abs(ifft2(fillimagecenter));
imshow(F,[]);


% --- Executes on mouse motion over figure - except title and menu.
function figure1_WindowButtonMotionFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



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
edit = str2num(get(handles.edit3,'string'));
newx = get(handles.edit4,'string');
newy = get(handles.edit7,'string');
set(handles.edit3,'max',100);
edit = [edit;str2num(newx),str2num(newy)];
set(handles.edit3,'string',num2str(edit));

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


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.edit3,'string','');

% --- Executes on mouse press over figure background, over a disabled or
% --- inactive control, or over an axes background.
function figure1_WindowButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[y,x]=ginput(1);
x=int64(x(1,1));
y=int64(y(1,1));
set(handles.edit4,'string',num2str(x));
set(handles.edit7,'string',num2str(y));



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
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
file = get(handles.edit1,'string');
img = imread(file);
[M,N] = size(img);
D0 = str2num(get(handles.edit8,'string'));
n = str2num(get(handles.edit9,'string'));
uv = str2num(get(handles.edit3,'string'));
uv(:,1) = uv(:,1)-M;
uv(:,2) = uv(:,2)-N;

for u = 1:2*M
    for v = 1:2*N
        H(u,v) = 1;
        for i = 1:length(uv)
            Dk(u,v) = [(u-M-uv(i,1))^2+(v-N-uv(i,2))^2]^(1/2);
            D_k(u,v) = [(u-M+uv(i,1))^2+(v-N+uv(i,2))^2]^(1/2);
            H(u,v) = H(u,v)*(1/(1+(D0/Dk(u,v))^2*n))*(1/(1+(D0/D_k(u,v))^2*n));
        end
    end
end
imshow(H);


% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
f = imread(file);
[M,N] = size(f);
F = fft2(f,2*M,2*N);
Fc = fftshift(F);
S = log(1+abs(Fc));

D0 = str2num(get(handles.edit8,'string'));
n = str2num(get(handles.edit9,'string'));
uv = str2num(get(handles.edit3,'string'));
uv(:,1) = uv(:,1)-M;
uv(:,2) = uv(:,2)-N;

for u = 1:2*M
    for v = 1:2*N
        H(u,v) = 1;
        for i = 1:length(uv)
            Dk(u,v) = [(u-M-uv(i,1))^2+(v-N-uv(i,2))^2]^(1/2);
            D_k(u,v) = [(u-M+uv(i,1))^2+(v-N+uv(i,2))^2]^(1/2);
            H(u,v) = H(u,v)*(1/(1+(D0/Dk(u,v))^2*n))*(1/(1+(D0/D_k(u,v))^2*n));
        end
    end
end
R = S.*H;
imshow(R,[]);

% --- Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
f = imread(file);
[M,N] = size(f);
F = fft2(f,2*M,2*N);
Fc = fftshift(F);
% S = log(1+abs(Fc));

[M,N] = size(f);
D0 = str2num(get(handles.edit8,'string'));
n = str2num(get(handles.edit9,'string'));
uv = str2num(get(handles.edit3,'string'));
uv(:,1) = uv(:,1)-M;
uv(:,2) = uv(:,2)-N;

for u = 1:2*M
    for v = 1:2*N
        H(u,v) = 1;
        for i = 1:length(uv)
            Dk(u,v) = [(u-M-uv(i,1))^2+(v-N-uv(i,2))^2]^(1/2);
            D_k(u,v) = [(u-M+uv(i,1))^2+(v-N+uv(i,2))^2]^(1/2);
            H(u,v) = H(u,v)*(1/(1+(D0/Dk(u,v))^2*n))*(1/(1+(D0/D_k(u,v))^2*n));
        end
    end
end
R = Fc.*H;
iR = real(ifft2(R));
for x = 1:2*M
    for y = 1:2*N
        h(x,y) = (-1)^(x+y);
    end
end
fillimage = h.*double(iR);
final = fillimage(1:M,1:N);
final = mat2gray(final);
imshow(final,[]);



% --- Executes on button press in pushbutton11.
function pushbutton11_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
f = imread(file);
[M,N] = size(f);
F = fft2(f,2*M,2*N);
Fc = fftshift(F);
S = log(1+abs(Fc));
S = mat2gray(S);
imshow(S,[]);


% --- Executes during object creation, after setting all properties.
function pushbutton8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in pushbutton12.
function pushbutton12_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
f = imread(file);
[M,N] = size(f);
d = str2num(get(handles.edit11,'string'));
flt = ones(size(2*M,2*N));
flt(:,M-d:M+d)=0;
flt(N-10:N+10,:)=1;
imshow(flt);


% --- Executes on button press in pushbutton13.
function pushbutton13_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
f = imread(file);
[M,N] = size(f);
F = fft2(f,2*M,2*N);
Fc = fftshift(F);
S = log(1+abs(Fc));
d = str2num(get(handles.edit11,'string'));
flt = ones(2*M,2*N);
flt(:,M-d:M+d)=0;
flt(N-10:N+10,:)=1;
R = S.*flt;
imshow(R,[]);


% --- Executes on button press in pushbutton14.
function pushbutton14_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = get(handles.edit1,'string');
f = imread(file);
[M,N] = size(f);
F = fft2(f,2*M,2*N);
Fc = fftshift(F);
% S = log(1+abs(Fc));
d = str2num(get(handles.edit11,'string'));
flt = ones(2*M,2*N);
flt(:,M-d:M+d)=0;
flt(N-10:N+10,:)=1;
R = Fc.*flt;

iR = real(ifft2(R));
for x = 1:2*M
    for y = 1:2*N
        h(x,y) = (-1)^(x+y);
    end
end
fillimage = h.*double(iR);
final = fillimage(1:M,1:N);
final = mat2gray(final);
imshow(final,[]);



function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit9_Callback(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit9 as text
%        str2double(get(hObject,'String')) returns contents of edit9 as a double


% --- Executes during object creation, after setting all properties.
function edit9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit11_Callback(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit11 as text
%        str2double(get(hObject,'String')) returns contents of edit11 as a double


% --- Executes during object creation, after setting all properties.
function edit11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


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
h=gcf; Experiment7; close(h);
