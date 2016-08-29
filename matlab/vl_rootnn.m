function root = vl_rootnn()
%VL_ROOTNN Get the root path of the MatConvNet toolbox.
%   VL_ROOTNN() returns the path to the MatConvNet toolbox.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.

% %mfilename returns a string containing the file name of the most recently invoked function. 
% When called from within the file, it returns the name of that file.
% This allows a function to determine its name, even if the file name has been changed.
% 
% %p = mfilename('fullpath') returns the full path and name of the file in which the call occurs,
% not including the filename extension.
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
root = fileparts(fileparts(mfilename('fullpath'))) ;
