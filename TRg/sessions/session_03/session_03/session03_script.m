	%%% ---[ 0. Loading datasets ]--- %%%
	run ../../matlab/vl_setupnn;

	% In this session, we are using subsets of MNIST
	% and CIFAR10 datasets in order to run several
	% epochs in short time using CPU.
	% 
	% These datasets can be pre-loaded as
	load('imdb_mnist_reduced');
	load('imdb_cifar10_reduced');

	
	%%% ---[ 1. Some small modifications to MatConvNet ]--- %%%
	%
	%	1.1. Update the file BatchNorm.m (located in session_03) 
	%	     in the folder matconvnet-1.0-beta17/matlab/+dagnn
	%
	%	2.2. Update the file cnn_train_dag.m (located in session_03) 
	%	     in the folder matconvnet-1.0-beta17/matlab


	%%% ---[ 2. Network initialization ]--- %%%
	%
	% The initialization of the network is one of the
	% most critical steps for training CNNs. This task
	% influences the convergence of the training and
	% imposes limits in the depth of the networks that
	% can be sucessfully trained.
	%
	% The main idea is to assign appropriate weights to each
	% convolution layer such that two things happen:
	%	1) The magnitude of the signal (i.e., mini batch of images)
	%	   remains within certain bounds (i.e., values are not too high
	%	   or too low) during the whole process (from the first to the 
	%	   last layer).
	%
	%	2) The gradient of the signal (i.e., mini batch of images)
	%	   remains within certain bounds too. In this case, as in any
	%	   optimization problem, the values of the gradient of X should
	%	   be like 1/1000 of the values of X. In other words, the magnitude
	%	   of the values of X should be around 3 orders of magnitude larger
	%	   than its gradient. Otherwise update steps would be...
	%	   Too large --> gradient exploding
	%	   Too short --> vanishing gradient
	%
	% To achieve these purposes there are some standard recipes for initialization
	%

	%  2.1. Initializing LeNet with constant factors
	%  	!Un-comment line #36 of lenet_train.m and run it with imdb_mnist_reduced dataset
	lenet_train(imdb_mnist_reduced, 'results/lenet_exp_01');

	%  2.2. Initializing LeNet with He's method
	%  	!Un-comment line #37 of lenet_train.m and run it with imdb_mnist_reduced dataset
	lenet_train(imdb_mnist_reduced, 'results/lenet_exp_02');	
	
	%  2.3. Initializing LeNet with Xavier's method
	%  	!Un-comment line #38 of lenet_train.m and run it with imdb_mnist_reduced dataset
	lenet_train(imdb_mnist_reduced, 'results/lenet_exp_03');	


	%%% ---[ 3. Using DropOut to improve generalization ]--- %%%

	%  3.1. Run LeNet with DropOut and check the difference w.r.t previous experiments
	%	Try changing the rate of drop too.
	lenet_train_drop(imdb_mnist_reduced, 'results/lenet_exp_04');

	%  3.2. Try to place dropout layers in different places of the net
	%       and check the difference w.r.t previous experiments.
	lenet_train_drop(imdb_mnist_reduced, 'results/lenet_exp_05');

	
	%%% ---[ 4. Using BatchNormalization to improve convergence and avoid vanishing gradient ]--- %%%

	%  4.1. Run AlexNet on imdb_cifar10_reduced to get a baseline to compare with
	alexnet_train(imdb_cifar10_reduced, 'results/alexnet_exp_01');

	%  4.2. Now run AlexNet with batch normalization on imdb_cifar10_reduced and compare
	%	with previous results
	alexnet_train_bnorm(imdb_cifar10_reduced, 'results/alexnet_exp_02');

	%  4.3. Try placing bnorm before ReLu layers and compare with previous results
	alexnet_train_bnorm(imdb_cifar10_reduced, 'results/alexnet_exp_03');

