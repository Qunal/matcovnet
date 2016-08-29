function imdb = createIMDB_RAM(folder)
	% imdb is a matlab struct with several fields, such as:
	%	- images: contains data, labels, ids dataset mean, etc.
	%	- meta: contains meta info useful for statistics and visualization
	%	- any other you want to add
	imdb = struct();

	% let's assume we have a folder with two
	% subfolders "cat" "nocat" containing images
	% for a binary problem
	positives = dir([folder '/cat/*.jpg']);% List of all files in folder
	negatives = dir([folder '/nocat/*.jpg']);
	imref = imread([folder '/cat/', positives(1).name]);
	[H, W, CH] = size(imref);

	% number of images
	NPos = numel(positives);
	NNeg = numel(negatives);
	N = NPos + NNeg;

	% we can initialize part of the structures already
	meta.sets = {'train', 'val'};
	meta.classes = {'nocat', 'cat'};

	% images go here
	images.data = zeros(H, W, CH, N, 'single');
	% this will contain the mean of the training set
	images.data_mean = zeros(H, W, CH, 'single');
	% a label per image
	images.labels = zeros(1, N);
	% vector indicating to which set an image belong, i.e.,
	% training, validation, etc.
	images.set = zeros(1, N);

	numImgsTrain = 0;
	% loading positive samples
	for i=1:numel(positives)
		im = single(imread([folder '/cat/', positives(i).name]));
		images.data(:,:,:, i) = im;
		images.labels(i) = 2;

		% in this case we select the set (train/val) randomly
		if(randi(10, 1) > 6)
			images.set(i) = 1;
			images.data_mean = images.data_mean + im;
			numImgsTrain = numImgsTrain + 1;
		else
			images.set(i) = 2;
		end
	end

	% loading negative samples
	for i=1:numel(negatives)
		im = single(imread([folder '/nocat/', negatives(i).name]));
		images.data(:,:,:, NPos+i) = im;
		images.labels(NPos+i) = 1;

		% in this case we select the set (train/val) randomly
		if(randi(10, 1) > 6)
			images.set(NPos+i) = 1;
			images.data_mean = images.data_mean + im;
			numImgsTrain = numImgsTrain + 1;
		else
			images.set(NPos+i) = 2;
		end
	end

	% let's finish to compute the mean
	images.data_mean = images.data_mean ./ numImgsTrain;

	% now let's add some randomization 
	indices = randperm(N);
	images.data(:,:,:,:) = images.data(:,:,:,indices);
	images.labels(:) = images.labels(indices);
	images.set(:) = images.set(indices);

	imdb.meta = meta;
	imdb.images = images;
	
end

