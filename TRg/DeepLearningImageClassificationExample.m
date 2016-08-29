function DeepLearningImageClassificationExample
% cnnMatFile = fullfile(pwd, 'imagenet-caffe-alex.mat')
% % Load MatConvNet network into a SeriesNetwork 
% convnet = helperImportMatConvNet(cnnMatFile)
%%
run  matlab/vl_setupnn
imagenetfile='data/imagenet-matconvnet-alex.mat';
%net = load(imagenetfile);
net = dagnn.DagNN.loadobj(load(imagenetfile));
net.mode = 'test' ;
%%
% net = vl_simplenn_tidy(net) ;

% load and preprocess an image% load and preprocess an image
im = imread('peppers.png') ;
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;
%% run the CNN
net.eval({'data', im_}) ;

% obtain the CNN otuput
scores = net.vars(net.getVarIndex('prob')).value ;
scores = squeeze(gather(scores)) ;

% show the classification results
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;