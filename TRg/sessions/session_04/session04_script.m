	%%% ---[ 1. Creating IMDBs from Images ]--- %%%
	run ../../matlab/vl_setupnn;

	%  1.1. IMDB when RAM memory is big enough
	imdb = createIMDB_RAM('./binaryDataset');
	
	%  1.2. Saving IMDB correctly
	save -v7.3 'imdb_cat' imdb;

	%  1.3. Testing IMDB with AlexNet
	alexnet_train_bnorm_binary(imdb, 'exp_binary_01a');

	%  1.4. Adding normalization to the IMDB
	imdbNorm = normalizeIMDB(imdb);

	%  1.4. Testing normalized IMDB with AlexNet
	alexnet_train_bnorm_binary(imdbNorm, 'exp_binary_01b');

	%  1.5. IMDB when RAM memory is NOT big enough
	imdb_disk = createIMDB_DISK('./binaryDataset');

	%  1.6. Testing IMDB with a new getBatchDisk function
	%
	%  Remember to change the getBatch function in line 69
	alexnet_train_bnorm_binary(imdb_disk, 'exp_binary_02');


	%%% ---[ 2. Fine-tuning from pre-trained models ]--- %%%

	%  2.1. Load a pre-trained version of AlexNet on ImageNet
	netPre = load('nets/imagenet-matconvnet-alex.mat');

	%  2.2. Use indbNorm to train AlexNet with no bnorm (baseline)
	alexnet_train_binary(imdbNorm, 'exp_binary_03');

	%  2.3. Use imdbNorm to fine-tune AlexNet to the given binary problem
	alexnet_train_binary(imdbNorm, 'exp_binary_04', netPre);

	%  2.4. Which method reaches best results earlier?

