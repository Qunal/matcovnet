function [net, info] = lenet_train(imdb, expDir)
% CNN_MNIST  Demonstrated MatConNet on MNIST using DAG
	run(fullfile(fileparts(mfilename('fullpath')), '../../', 'matlab', 'vl_setupnn.m')) ;

	% some common options
	opts.train.batchSize = 100 ;
	opts.train.numEpochs = 60 ;
	opts.train.continue = true ;
	opts.train.gpus = [] ;
	opts.train.learningRate = 0.001 ;
	opts.train.expDir = expDir;
	opts.train.numSubBatches = 1 ;
	% getBatch options
	bopts.useGpu = numel(opts.train.gpus) >  0 ;


	% network definition!
	% MATLAB handle, passed by reference
	net = dagnn.DagNN() ;
	net.addLayer('conv1', dagnn.Conv('size', [5 5 1 20], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'input'}, {'conv1'},  {'conv1f'  'conv1b'});
	net.addLayer('pool1', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 0 0 0]), {'conv1'}, {'pool1'}, {});

	net.addLayer('conv2', dagnn.Conv('size', [5 5 20 50], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool1'}, {'conv2'},  {'conv2f'  'conv2b'});
	net.addLayer('pool2', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2 2], 'pad', [0 0 0 0]), {'conv2'}, {'pool2'}, {});

	net.addLayer('conv3', dagnn.Conv('size', [4 4 50 500], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool2'}, {'conv3'},  {'conv3f'  'conv3b'});
	net.addLayer('relu3', dagnn.ReLU(), {'conv3'}, {'relu3'}, {});

	net.addLayer('classifier', dagnn.Conv('size', [1 1 500 10], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu3'}, {'classifier'},  {'conv4f'  'conv4b'});
	net.addLayer('prediction', dagnn.SoftMax(), {'classifier'}, {'prediction'}, {});
	net.addLayer('objective', dagnn.Loss('loss', 'log'), {'prediction', 'label'}, {'objective'}, {});
	net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'prediction','label'}, 'error') ;
	% -- end of the network

	% initialization of the weights (CRITICAL!!!!)
	%initNet(net, 1/100*ones(1,4), 0*ones(1,4));
	%initNet_He(net);
	initNet_xavier(net);

	% do the training!
	info = cnn_train_dag(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train, 'val', find(imdb.images.set == 3)) ;
end

function initNet_He(net, f)
	net.initParams();
	for l=1:length(net.layers)
		% is a convolution layer?
		if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);

			[h,w,in,out] = size(net.params(f_ind).value);
			he_gain = 0.3*sqrt(2/(h*w*in)); % sqrt(1/fan_in)
			net.params(f_ind).value = he_gain*randn(size(net.params(f_ind).value), 'single');
			net.params(f_ind).learningRate = 1;
			net.params(f_ind).weightDecay = 1;

			net.params(b_ind).value = zeros(size(net.params(b_ind).value), 'single');
			net.params(b_ind).learningRate = 0.5;
			net.params(b_ind).weightDecay = 1;
		end
	end
end

function initNet_xavier(net)
	net.initParams();
	for l=1:length(net.layers)
		% is a convolution layer?
		if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);

			[h,w,in,out] = size(net.params(f_ind).value);
			xavier_gain = 0.5*sqrt(2/(h*w*out));
			net.params(f_ind).value = xavier_gain*randn(size(net.params(f_ind).value), 'single');
			net.params(f_ind).learningRate = 1;
			net.params(f_ind).weightDecay = 1;

			net.params(b_ind).value = zeros(size(net.params(b_ind).value), 'single');
			net.params(b_ind).learningRate = 0.5;
			net.params(b_ind).weightDecay = 1;
		end
	end
end

function initNet(net, F, B)
	net.initParams();
	%
	i = 1;
	for l=1:length(net.layers)
		% is a convolution layer?
		if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);

			net.params(f_ind).value = F(i)*randn(size(net.params(f_ind).value), 'single');
			net.params(f_ind).learningRate = 1;
			net.params(f_ind).weightDecay = 1;

			net.params(b_ind).value = B(i)*randn(size(net.params(b_ind).value), 'single');
			net.params(b_ind).learningRate = 0.5;
			net.params(b_ind).weightDecay = 1;
			i = i + 1;
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
