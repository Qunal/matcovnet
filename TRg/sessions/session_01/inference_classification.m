function inference_classification(im, net)
	im_ = single(im) ; % note: 0-255 range
	im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
	im_ = im_ - net.meta.normalization.averageImage;

	% run the CNN
	net.eval({'input', im_});

	% obtain the CNN otuput
	scores = net.vars(net.getVarIndex('prob')).value;
	scores = squeeze(gather(scores));

	% show the classification results
	[bestScore, best] = max(scores);
	figure(1) ; clf ; imagesc(im);
	title(sprintf('%s (%d), score %.3f', net.meta.classes.description{best}, best, bestScore));
end
