function visualizeFilters(net, epoch, lpath)
    run('../deps/vlfeat-0.9.20/toolbox/vl_setup');
    
    % for each layer...
    NSQ = 100;
    for l=1:length(net.layers)
	% is a convolution layer?
	if(strcmp(class(net.layers(l).block), 'dagnn.Conv'))
		% visualize it!
		index = net.layers(l).paramIndexes(1);
		F = net.params(index).value;
		[H, W, C, K] = size(F);
    		for c=1:C
			for k=1:K
				tmp = F(:,:,c, k);
				dataMin = min(tmp(:));
				dataMax = max(tmp(:));

				a = 1.0 / (dataMax - dataMin + eps) ;
				F(:,:,c, k) = (F(:,:,c, k) - dataMin)*a;
			end
		end

		if(size(F, 3) == 3)
			process_three_channel_tensor(F, epoch, l, NSQ, lpath);
		else
			F = reshape(F, size(F, 1), size(F, 2), size(F, 3)*size(F, 4));
			process_one_channel_tensor(F, epoch, l, NSQ, lpath);
		end
	end
	
    end
end

function process_one_channel_tensor(F, epoch, layer, NSQ, lpath)
        for batch=1:ceil(size(F, 3)/NSQ)
            ind1 = 1+((batch-1)*NSQ);
            ind2 = min(size(F, 3), batch*NSQ);
            
            JJ = vl_imarray(F(:, :, ind1:ind2),'spacing',1);
            JJ = imresize(JJ, [800, 800]);
            
	    dir0 = sprintf('%s/layer_%d/fb_%d', lpath, layer, batch);
	    if (~exist(dir0,'dir'))
		mkdir(dir0);
	    end
	    str = sprintf('%s/epoch_%d.png', dir0, epoch);
	    imwrite(JJ, str);
        end
end

function process_three_channel_tensor(F, epoch, layer, NSQ, lpath)
        for batch=1:ceil(size(F, 4)/NSQ)
            ind1 = 1+((batch-1)*NSQ);
            ind2 = min(size(F, 4), batch*NSQ);
            
            JJ = vl_imarray(F(:, :, :, ind1:ind2),'spacing',1);
            JJ = imresize(JJ, [800, 800]);
            
	    dir0 = sprintf('%s/layer_%d/fb_%d', lpath, layer, batch);
	    if (~exist(dir0,'dir'))
		mkdir(dir0);
	    end
	    str = sprintf('%s/epoch_%d.png', dir0, epoch);
	    imwrite(JJ, str);
        end
end

function [b] = isConvLayer(layer)
	b = false;
	if(strcmp(layer.type, 'conv')) 
		[H, W, L, M] = size(layer.weights{1});
		if( (H~=1) || (W~=1) )
			b = true;
		end
	end
end
