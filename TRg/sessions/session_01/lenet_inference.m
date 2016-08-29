function [class, score] = lenet_inference(im, net, conserveMemory)
	% single image that will require normalization, right?

 	im_ = single(im) ; % note: 0-255 range
	%im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
	%im_ = im_ - net.meta.normalization.averageImage;
	
	% run the CNN
	net.conserveMemory = conserveMemory;
	net.eval({'input', im_});

	% obtain the CNN otuput
	scores = net.vars(net.getVarIndex('prediction')).value;
	scores = squeeze(gather(scores));

	% show the classification results
	[score, class] = max(scores);
	figure() ; clf ; imagesc(im);
	title(sprintf('%s (%d), score %.3f', net.meta.classes.description{class}, class, score));
end
