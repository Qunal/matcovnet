%%% ---[ 1. First Contact ]--- %%%
	run ../../matlab/vl_setupnn;

	% loading a pre-trained Net, i.e., AlexNet
	alexNet = load('TRg/sessions/Nets/imagenet-alex.mat');

	% keep the classes name, that in this case are from ILSVRC2012
	classes = alexNet.classes.description;
	% convert the net from a standard Matlab structure to a dagnn object
	alexNet = dagnn.DagNN.fromSimpleNN(alexNet, 'canonicalNames', true);
	% assign classes back to the dagnn object (useful to perform inference later)
	alexNet.meta.classes.description = classes;

	% check net members of the net. Get familiar with these fields
	alexNet

	% Visualizing net layers and properties
	alexNet.print('MaxNumColumns', 5);

	% filter visualization (output on directory results/filters_alexnet)
	visualizeFilters(alexNet, 1, 'results/filters_alexnet');

	% Perform inference with alexNet.
	im = imread('../data/dog.jpg');
	inference_classification(im, alexNet); 


%%% ---[ 2. LeNet Training ]--- %%%

	% loading the dataset MNIST as an IMDB structure
	imdb_mnist = load('../data/imdb_mnist.mat');

	% we start training! partial results will go to results/lenet_experiment_1 folder
	[net_lenet, info] = lenet_train(imdb_mnist, 'results/lenet_experiment_1');

	% let's check together the training code to have some fun :)

%%% ---[ 3. LeNet Inference ]--- %%%

	% load a previously trained (LeNet) net
	lenet1 = load('../Nets/lenet.mat');

	% make it a Dag object
	lenet1 = dagnn.DagNN.loadobj(lenet1.net);
	% again, keep the meta info about the classes (useful for inference)
	lenet1.meta.classes.description = imdb_mnist.meta.classes;

	% let's do some inference over MNIST
	% first we select validation images (those images belonging to set == 3)
	val_images = find(imdb_mnist.images.set == 3);
	% Let's pick a random image from validation set
	i = 5; % change i to try with different images
	% we now infer the result using LeNet
	[class, score] = lenet_inference(imdb_mnist.images.data(:,:,:,val_images(i)), lenet1, false);
	str = sprintf('I think your number is a...[%s!]', imdb_mnist.meta.classes{class}); disp(str);

	% Since we indicated we want to keep all the intermediate results in the previous inference
	% i.e., the last argument set to "false", now we can check the intermediate activations!

	lenet1.vars(2) % This corresponds to the first convolution layer
	% we now visualize the activations of the first and second filter of the first convolution
	figure; imagesc(lenet1.vars(2).value(:,:,1));
	figure; imagesc(lenet1.vars(2).value(:,:,2));

%%% ---[ AlexNet Training ]--- %%%
	% load CIFAR-10
	imdb_cifar10 = load('../data/imdb_cifar10.mat');

	% perform training using AlexNet architecture
	[net_alexnet, info] = alexnet_train(imdb_cifar10, 'results/cifar_10_experiment_1');

	% let's check together the training code to have some fun :)

