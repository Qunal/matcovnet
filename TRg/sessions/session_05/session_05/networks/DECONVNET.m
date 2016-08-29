function [net, info] = DECONVNET(imdb, netF, inpt, varargin)

	% some common options
	trainer = @cnn_train_dag_seg;

	opts.train.extractStatsFn = @extract_stats_segmentation;
	opts.train.batchSize = 7;
	opts.train.numEpochs = 60 ;
	opts.train.continue = true ;
	opts.train.gpus = [] ;
	opts.train.learningRate = [1e-1*ones(1, 10),  1e-2*ones(1, 5)];
	opts.train.weightDecay = 1e-4;
	opts.train.momentum = 0.9;
	opts.train.expDir = inpt.expDir;
	opts.train.savePlots = false;
	opts.train.numSubBatches = 5;
	% getBatch options
	bopts.useGpu = numel(opts.train.gpus) >  0 ;

	% organize data
	K = 2; % how many examples per domain	
	trainData = find(imdb.images.set == 1);
	valData = find(imdb.images.set == 2);
	
	% debuging code
	opts.train.exampleIndices = [trainData(randperm(numel(trainData), K)), valData(randperm(numel(valData), K))];

	opts.train.classesNames = {'sky', 'building', 'road', 'sidewalk', 'fence', 'vegetation', 'pole', 'car', 'sign', 'pedestrian', 'cyclist'};
	colorMap  = (1/255)*[		    
					    128 128 128
					    128 0 0
					    128 64 128
					    0 0 192
					    64 64 128
					    128 128 0
					    192 192 128
					    64 0 128
					    192 128 128
					    64 64 0
					    0 128 192
					    ];
	opts.train.colorMapGT = [0 0 0; colorMap];
	opts.train.colorMapEst = colorMap;

	% network definition
	net = dagnn.DagNN() ;
	net.addLayer('conv1', dagnn.Conv('size', [7 7 3 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'input'}, {'conv1'},  {'conv1f'  'conv1b'});
	net.addLayer('bn1', dagnn.BatchNorm('numChannels', 64), {'conv1'}, {'bn1'}, {'bn1f', 'bn1b', 'bn1m'});
	net.addLayer('relu1', dagnn.ReLU(), {'bn1'}, {'relu1'}, {});
	net.addLayer('pool1', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu1'}, {'pool1'}, {});

	net.addLayer('conv2', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'pool1'}, {'conv2'},  {'conv2f'  'conv2b'});
	net.addLayer('bn2', dagnn.BatchNorm('numChannels', 64), {'conv2'}, {'bn2'}, {'bn2f', 'bn2b', 'bn2m'});
	net.addLayer('relu2', dagnn.ReLU(), {'bn2'}, {'relu2'}, {});
	net.addLayer('pool2', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu2'}, {'pool2'}, {});

	net.addLayer('deconv3', dagnn.ConvTranspose('size', [7 7 64 64], 'hasBias', true, 'upsample', [2,2], 'crop', [3 2 3 2]), {'pool2'}, {'deconv3'},  {'deconv3f'  'deconv3b'});
	net.addLayer('bn3', dagnn.BatchNorm('numChannels', 64), {'deconv3'}, {'bn3'}, {'bn3f', 'bn3b', 'bn3m'});
	net.addLayer('relu3', dagnn.ReLU(), {'bn3'}, {'relu3'}, {});


	net.addLayer('deconv4', dagnn.ConvTranspose('size', [7 7 64 64], 'hasBias', true, 'upsample', [2,2], 'crop', [2 2 3 2]), {'relu3'}, {'deconv4'},  {'deconv4f'  'deconv4b'});
	net.addLayer('bn4', dagnn.BatchNorm('numChannels', 64), {'deconv4'}, {'bn4'}, {'bn4f', 'bn4b', 'bn4m'});
	net.addLayer('relu4', dagnn.ReLU(), {'bn4'}, {'relu4'}, {});

	net.addLayer('classifier', dagnn.Conv('size', [1 1 64 11], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu4'}, {'classifier'},  {'classf'  'classb'});
	net.addLayer('prob', dagnn.SoftMax(), {'classifier'}, {'prob'}, {});
	net.addLayer('objective', dagnn.LossSemantic('weights', true), {'prob','label'}, 'objective');
	% -- end of the network

	% do the training!
	initNet(net, 1e-2*ones(1, 5), 1e-2*ones(1, 5));
	net.conserveMemory = false;

	info = trainer(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train, 'train', trainData, 'val', valData) ;
end


% function on charge of creating a batch of images + labels
function inputs = getBatch(opts, imdb, batch)
	images = imdb.images.data(:,:,:,batch) ;
	labels = imdb.images.labels(:, :, :, batch) ;
	if opts.useGpu > 0
  		images = gpuArray(images);
		labels = gpuArray(labels);
	end

	inputs = {'input', images, 'label', labels} ;
end

function initNet(net, F, B)
	net.initParams();
	%
	i = 1;
	for l=1:length(net.layers)
		% is a convolution layer?
		if(strcmp(class(net.layers(l).block), 'dagnn.Conv') || strcmp(class(net.layers(l).block), 'dagnn.ConvTranspose'))
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
