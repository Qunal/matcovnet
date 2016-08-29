function [net, info] = alexnet_train(imdb, expDir)
%  Demonstrated MatConNet on CIFAR-10 using DAG
	run(fullfile(fileparts(mfilename('fullpath')), '../../', 'matlab', 'vl_setupnn.m')) ;

	% some common options
	opts.train.batchSize = 100;
	opts.train.numEpochs = 20 ;
	opts.train.continue = true ;
	opts.train.gpus = [1] ;
	opts.train.learningRate = [1e-1*ones(1, 10),  1e-2*ones(1, 5)];
	opts.train.weightDecay = 3e-4;
	opts.train.momentum = 0.;
	opts.train.expDir = expDir;
	opts.train.numSubBatches = 1;
	% getBatch options
	bopts.useGpu = numel(opts.train.gpus) >  0 ;


	% network definition!
	% MATLAB handle, passed by reference
	net = dagnn.DagNN() ;

	% special padding for CIFAR-10
	net.addLayer('conv1', dagnn.Conv('size', [11 11 3 96], 'hasBias', true, 'stride', [4, 4], 'pad', [20 20 20 20]), {'input'}, {'conv1'},  {'conv1f'  'conv1b'});
	net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'relu1'}, {});
	net.addLayer('lrn1', dagnn.LRN('param', [5 1 2.0000e-05 0.7500]), {'relu1'}, {'lrn1'}, {});
	net.addLayer('pool1', dagnn.Pooling('method', 'max', 'poolSize', [3, 3], 'stride', [2 2], 'pad', [0 0 0 0]), {'lrn1'}, {'pool1'}, {});

	net.addLayer('conv2', dagnn.Conv('size', [5 5 48 256], 'hasBias', true, 'stride', [1, 1], 'pad', [2 2 2 2]), {'pool1'}, {'conv2'},  {'conv2f'  'conv2b'});
	net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'relu2'}, {});
	net.addLayer('lrn2', dagnn.LRN('param', [5 1 2.0000e-05 0.7500]), {'relu2'}, {'lrn2'}, {});
	net.addLayer('pool2', dagnn.Pooling('method', 'max', 'poolSize', [3, 3], 'stride', [2 2], 'pad', [0 0 0 0]), {'lrn2'}, {'pool2'}, {});

	net.addLayer('conv3', dagnn.Conv('size', [3 3 256 384], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'pool2'}, {'conv3'},  {'conv3f'  'conv3b'});
	net.addLayer('relu3', dagnn.ReLU(), {'conv3'}, {'relu3'}, {});

	net.addLayer('conv4', dagnn.Conv('size', [3 3 192 384], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'relu3'}, {'conv4'},  {'conv4f'  'conv4b'});
	net.addLayer('relu4', dagnn.ReLU(), {'conv4'}, {'relu4'}, {});
	
	net.addLayer('conv5', dagnn.Conv('size', [3 3 192 256], 'hasBias', true, 'stride', [1, 1], 'pad', [1 1 1 1]), {'relu4'}, {'conv5'},  {'conv5f'  'conv5b'});
	net.addLayer('relu5', dagnn.ReLU(), {'conv5'}, {'relu5'}, {});
	net.addLayer('pool5', dagnn.Pooling('method', 'max', 'poolSize', [3 3], 'stride', [2 2], 'pad', [0 0 0 0]), {'relu5'}, {'pool5'}, {});

	net.addLayer('fc6', dagnn.Conv('size', [1 1 256 4096], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool5'}, {'fc6'},  {'conv6f'  'conv6b'});
	net.addLayer('relu6', dagnn.ReLU(), {'fc6'}, {'relu6'}, {});

	net.addLayer('fc7', dagnn.Conv('size', [1 1 4096 4096], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu6'}, {'fc7'},  {'conv7f'  'conv7b'});
	net.addLayer('relu7', dagnn.ReLU(), {'fc7'}, {'relu7'}, {});

	net.addLayer('classifier', dagnn.Conv('size', [1 1 4096 10], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu7'}, {'classifier'},  {'conv8f'  'conv8b'});
	net.addLayer('prob', dagnn.SoftMax(), {'classifier'}, {'prob'}, {});
	net.addLayer('objective', dagnn.Loss('loss', 'log'), {'prob', 'label'}, {'objective'}, {});
	net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'prob','label'}, 'error') ;
	% -- end of the network

	% initialization of the weights (CRITICAL!!!!)
	initNet(net, 1/100);

	% do the training!
	info = cnn_train_dag(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train, 'val', find(imdb.images.set == 3)) ;
end

function initNet(net, f)
	net.initParams();
	%

	f_ind = net.layers(1).paramIndexes(1);
	b_ind = net.layers(1).paramIndexes(2);
	net.params(f_ind).value = 10*f*randn(size(net.params(f_ind).value), 'single');
	net.params(f_ind).learningRate = 1;
	net.params(f_ind).weightDecay = 1;

	for l=2:length(net.layers)
		% is a convolution layer?
		if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);

			[h,w,in,out] = size(net.params(f_ind).value);
			net.params(f_ind).value = f*randn(size(net.params(f_ind).value), 'single');
			net.params(f_ind).learningRate = 1;
			net.params(f_ind).weightDecay = 1;

			net.params(b_ind).value = f*randn(size(net.params(b_ind).value), 'single');
			net.params(b_ind).learningRate = 0.5;
			net.params(b_ind).weightDecay = 1;
		end
	end
end

% function on charge of creating a batch of images + labels
function inputs = getBatch(opts, imdb, batch)
	images = imdb.images.data(:,:,:,batch) ;
	labels = imdb.images.labels(1,batch) ;
	if opts.useGpu > 0
  		images = gpuArray(images) ;
	end

	inputs = {'input', images, 'label', labels} ;
end
