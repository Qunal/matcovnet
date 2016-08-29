%% training
% load model
% Dir structure is confusing
% for runing some matcovnet\examples\mist\data
% for others matcovnet\data\mnist
%models      matcovnet\data\models

modeldir='data/models/'
modelfile='imagenet-matconvnet-alex.mat';
file=strcat(modeldir,modelfile);
net=load(file);
% net rather than alexnet makes subsequent code generic
%% Insect model
% meta holds  input size, train opts labels and sizes 
net.meta.classes % 1000 lables ILSVCR2012 data
net.meta.classes.description{1}
% Copy descriptions for future 
classes=net.meta.classes.description;
% layers are heart and pretraiend model includes weights
net.layers(1)% params  pointers to  filter/weights and biases for layer
%weights are held in params and can be shared across layers
size(net.params(2).value)% HXWXDXN HghtWidth,Depth/ChannelX Batch 11X11X3X96
min(net.params(2).value)
max(net.params(2).value)
mean(net.params(2).value)
% blocks
net.layers(1).block% Size of layer HWDN and Padding, stride and options like CUDA size 11X11X3X96
%% visualize
% Visulaize net
%print(alexnet,'MaxNumColumns',7)
%visualizeFilters(alexnet,1,'results/filters_alexnet')
% anet=dagnn.DagNN.fromSimpleNN(alexnet,'canonicalNames',true); 
imshow( net.layers{1}.filters(:, :, 3, 1), [] ) ;% For SimpleNN
% net.layers(1).params{1,1}%Filter conv1ffor DAG, Layers has a pointer to shared weights in params
%net.layers(1).params{1,1}%Biases conv1b
imshow(net.params(1).value(:,:,3,1),[]);%  show 96 of plots 

%% Inference 25 minute in video
img=imread('dog');
img=imresize(img,net.meta.normalization.imazeSize(1:2));% 227X227X3
img=img-net.meta.normalization.averageImage;% subtract mean
inference_classification(net,img);
% alreadyDagNN