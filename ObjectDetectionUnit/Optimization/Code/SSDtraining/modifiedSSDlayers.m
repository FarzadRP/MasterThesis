function ssdLayerGraph = modifiedSSDlayers(imageSize, numClasses, baseNetwork, varargin)
%ssdLayers Create a SSD object detection network
%
%   This function requires the Deep Learning Toolbox.
%
%   Single Shot Detector (SSD) is a convolutional neural network based
%   object detector which predicts bounding box coordinates, classification
%   scores, and associated class labels.
%
%   lgraph = ssdLayers(imageSize, numClasses, baseNetwork) returns an SSD
%   object detection network as a layerGraph object. The baseNetwork
%   parameter can be specified as a string, including 'vgg16', 'resnet50',
%   and 'resnet101'.
%
%   lgraph = ssdLayers(imageSize, numClasses, baseNetwork, anchorBoxes,
%   predictorLayerNames) returns an SSD object detection network with
%   custom anchor boxes specified using anchorBoxes, connected to the
%   network layers at locations specified by predictorLayerNames.
%   baseNetwork must be a LayerGraph, SeriesNetwork, or a DAGNetwork
%   object.
%
%   Inputs
%   ------
%   imageSize    - size of the input image specified as a vector
%                  [H W] or [H W C], where H and W are the image height and
%                  width, and C is the number of image channels.
%
%   numClasses   - A positive scalar that specifies the number of classes
%                  the network should be configured to classify.
%
%   baseNetwork  - A pretrained classification network that forms the
%                  basis of the SSD network. Specify the network as a 
%                  SeriesNetwork, a DAGNetwork, a LayerGraph, or by name.
%
%                  Valid network names are listed below, and require
%                  installation of the associated Add-On:
%
%                  <a href="matlab:helpview('deeplearning','vgg16')">'vgg16'</a>
%                  <a href="matlab:helpview('deeplearning','resnet50')">'resnet50'</a>
%                  <a href="matlab:helpview('deeplearning','resnet101')">'resnet101'</a>
%
%                  <a href="matlab:helpview('deeplearning','pretrained_networks')">Pretrained networks supported in MATLAB.</a>
%
%   anchorBoxes  - A 1-by-M cell array, each containing an K-by-2 matrix 
%                  defining the [height width] of K anchor boxes. Anchor 
%                  boxes are determined based on a priori knowledge of the 
%                  scale and aspect ratio of objects in the training 
%                  dataset. K can be different for each cell element. 
%
%   predictorLayerNames - An M-element vector of strings or an M-element
%                         cell array of character vectors specifying the
%                         names of layers in input network. The SSD
%                         detection sub-networks are attached to these
%                         layers.
%
% Example 1: Create an SSD network with VGG-16.
% ---------------------------------------------
% % Specify the base network.
% baseNetwork = 'vgg16';
%
% % Specify the image size.
% imageSize = [300 300 3];
%
% % Specify the number of classes to detect.
% numClasses = 2;
%
% % Create the SSD object detection network.
% lgraph = ssdLayers(imageSize, numClasses, baseNetwork);
%
% % Visualize the network using the network analyzer.
% analyzeNetwork(lgraph)
%
% Example 2: Create an SSD network with SqueezeNet.
% -------------------------------------------------
% % Specify the base network.
% baseNetwork = squeezenet;
%
% % Specify the image size.
% imageSize = [300 300 3];
%
% % Specify the number of classes to detect.
% numClasses = 2;
%
% % Name of the layer to connect the detection sub-network.
% layersToConnect =  "fire9-concat"; 
%
% anchorBoxes = {[30 60; 60 30; 50 50; 100 100]};
%
% % Create the SSD object detection network.
% lgraph = ssdLayers(imageSize, numClasses, baseNetwork, anchorBoxes, layersToConnect);
%
% % Visualize the network using the network analyzer.
% analyzeNetwork(lgraph)
%
% See also trainSSDObjectDetector, anchorBoxLayer, focalLossLayer, 
%          ssdMergeLayer, analyzeNetwork.

%   References
%   ----------
%   [1] Liu, Wei, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
%       Scott Reed, Cheng Yang Fu, and Alexander C. Berg. "SSD: Single shot
%       multibox detector." In 14th European Conference on Computer Vision,
%       ECCV 2016. Springer Verlag, 2016.
%
%   [2] Huang, Jonathan, et al. "Speed/accuracy trade-offs for modern
%       convolutional object detectors." IEEE CVPR 2017.

% Copyright 2019 The MathWorks, Inc.

    vision.internal.requiresNeuralToolbox(mfilename);

    narginchk(3, 5);
    [baseNetwork, imageSize, numClasses] = iParseInputs(imageSize, numClasses, baseNetwork);

    if isstring(baseNetwork) || ischar(baseNetwork)
        narginchk(3,3);
        switch(baseNetwork)
            case 'vgg16'
                net = vgg16();
                ssdLayerGraph = createFromVGG16(net, imageSize, numClasses);
            case 'resnet50'
                net = resnet50();
                ssdLayerGraph = createFromResnet50(net, imageSize, numClasses);
            case 'resnet101'
                net = resnet101();
                ssdLayerGraph = createFromResnet101(net, imageSize, numClasses);
        end
    else
        % Build a custom network.
        narginchk(5,5);
        [predictorBranchNames, anchorBoxes] = iParseOptionalInputs(varargin{:});
        ssdLayerGraph = createFromCustomNetwork(baseNetwork, imageSize, numClasses, predictorBranchNames, anchorBoxes); 
    end

    analysis = nnet.internal.cnn.analyzer.NetworkAnalyzer(ssdLayerGraph);
    analysis.applyConstraints();
    try
        analysis.throwIssuesIfAny();
    catch ME
        throwAsCaller(ME);
    end
    ssdLayerGraph = analysis.LayerGraph;
end
%--------------------------------------------------------------------------
function ssdLayerGraph = createFromVGG16(lgraph, inputImageSize, numClasses)
    % lgraph is a SeriesNetwork, and so need to extract Layers from lgraph
    % to construct a layerGraph.
    lgraph = layerGraph(lgraph.Layers);
    lgraph = iReplaceImageInputLayer(lgraph, inputImageSize);

    weightsInitializerValue = 'glorot';
    biasInitializerValue = 'zeros';
    baseLayers = iUpdateLastLayersVGG16(lgraph.Layers, weightsInitializerValue, biasInitializerValue);
    extraLayers = iCreateExtraLayers(weightsInitializerValue, biasInitializerValue);

    % Create predictor layers.
    layersToConnect = ["conv4_3", "relu7", "relu6_2", "relu7_2", "relu8_2", "relu9_2"];
    
    % Compute min and max size of anchor boxes based on input image size.
    numAnchorBoxLayers = size(layersToConnect,2);
    [minAnchorSize, maxAnchorSize] = iComputeAnchorBoxeSizes(inputImageSize,numAnchorBoxLayers );

    % The following will change based on how anchorBoxLayer is parameterized.
    anchorBoxes = cell(numel(layersToConnect), 1);    
    anchorBoxes{1} = iMakeAnchorBoxes(maxAnchorSize(1,1), minAnchorSize(1,1), 2);
    anchorBoxes{2} = iMakeAnchorBoxes(maxAnchorSize(1,2), minAnchorSize(1,2), [3, 2]);
    anchorBoxes{3} = iMakeAnchorBoxes(maxAnchorSize(1,3), minAnchorSize(1,3), [3, 2]);
    anchorBoxes{4} = iMakeAnchorBoxes(maxAnchorSize(1,4), minAnchorSize(1,4), [3, 2]);
    anchorBoxes{5} = iMakeAnchorBoxes(maxAnchorSize(1,5), minAnchorSize(1,5), 2);
    anchorBoxes{6} = iMakeAnchorBoxes(maxAnchorSize(1,6), minAnchorSize(1,6), 2);

    ssdLayerGraph = iCreateSSDLayerGraph(baseLayers, extraLayers, layersToConnect, ...
                         weightsInitializerValue, biasInitializerValue, ...
                         numClasses, ...
                         anchorBoxes);
end
%--------------------------------------------------------------------------
function ssdLayerGraph = createFromResnet50(net, inputImageSize, numClasses)

    % Construction of the network using a resnet50 backbone is described
    % in the following paper. See section 3.6.3, under resnet101:
    %
    % We use the feature map from the last layer of the “conv4” block.  When
    % operating in atrous mode, the stride size is 8 pixels, otherwise it
    % is 16 pixels. Five additional convolutional layers with decaying
    % spatial resolution are appended, which have depths 512, 512, 256,
    % 256, 128, respectively.  We have experimented with including the
    % feature map from the last layer of the “conv5” block. With “conv5”
    % features, them AP numbers are very similar, but the computational
    % costs are higher. Therefore we choose to use the last layer of the
    % “conv4” block. During  training, a  base learning rate of 3e-4 is
    % used. We use a learning rate warm up strategy similar to the VGG one.
    %
    % Huang, Jonathan, et al. "Speed/accuracy trade-offs for modern
    %       convolutional object detectors." IEEE CVPR 2017. 
    lgraph = layerGraph(net);
    lgraph = iReplaceImageInputLayer(lgraph, inputImageSize);

    weightsInitializerValue = 'glorot';
    biasInitializerValue = 'zeros';

    % Drop bottom layers from resnet50
    baseLayers = iRemoveLayers(lgraph, 'activation_40_relu');
    extraLayers = iCreateExtraLayers(weightsInitializerValue, biasInitializerValue);

    % Create predictor layers.
    layersToConnect = ["activation_22_relu", "activation_40_relu", "relu6_2", "relu7_2", "relu8_2", "relu9_2"];
    
    % Compute min and max size of anchor boxes based on input image size.
    numAnchorBoxLayers = size(layersToConnect,2);
    [minAnchorSize, maxAnchorSize] = iComputeAnchorBoxeSizes(inputImageSize,numAnchorBoxLayers );
    
    % The following will change based on how anchorBoxLayer is parameterized.
    anchorBoxes = cell(numel(layersToConnect), 1);
    anchorBoxes{1} = iMakeAnchorBoxes(maxAnchorSize(1,1), minAnchorSize(1,1), 2);
    anchorBoxes{2} = iMakeAnchorBoxes(maxAnchorSize(1,2), minAnchorSize(1,2), [3, 2]);
    anchorBoxes{3} = iMakeAnchorBoxes(maxAnchorSize(1,3), minAnchorSize(1,3), [3, 2]);
    anchorBoxes{4} = iMakeAnchorBoxes(maxAnchorSize(1,4), minAnchorSize(1,4), [3, 2]);
    anchorBoxes{5} = iMakeAnchorBoxes(maxAnchorSize(1,5), minAnchorSize(1,5), 2);
    anchorBoxes{6} = iMakeAnchorBoxes(maxAnchorSize(1,6), minAnchorSize(1,6), 2);

    ssdLayerGraph = iCreateSSDLayerGraph(baseLayers, extraLayers, layersToConnect, ...
                         weightsInitializerValue, biasInitializerValue, ...
                         numClasses, ...
                         anchorBoxes);
end
%--------------------------------------------------------------------------
function ssdLayerGraph = createFromResnet101(net, inputImageSize, numClasses)

    % Construction of the network using a resnet101 backbone is described
    % in the following paper. See section 3.6.3, under resnet101:
    %
    % We use the feature map from the last layer of the “conv4” block.  When
    % operating in atrous mode, the stride size is 8 pixels, otherwise it
    % is 16 pixels. Five additional convolutional layers with decaying
    % spatial resolution are appended, which have depths 512, 512, 256,
    % 256, 128, respectively.  We have experimented with including the
    % feature map from the last layer of the “conv5” block. With “conv5”
    % features, them AP numbers are very similar, but the computational
    % costs are higher. Therefore we choose to use the last layer of the
    % “conv4” block. During  training, a  base learning rate of 3e-4 is
    % used. We use a learning rate warm up strategy similar to the VGG one.
    %
    % Huang, Jonathan, et al. "Speed/accuracy trade-offs for modern
    %       convolutional object detectors." IEEE CVPR 2017. 
    
    lgraph = layerGraph(net);
    lgraph = iReplaceImageInputLayer(lgraph, inputImageSize);

    weightsInitializerValue = 'glorot';
    biasInitializerValue = 'zeros';

    % Drop bottom layers from resnet101
    baseLayers = iRemoveLayers(lgraph, 'res4b22_relu');
    extraLayers = iCreateExtraLayers(weightsInitializerValue, biasInitializerValue);

    % Create predictor layers.
    layersToConnect = ["res3b3_relu", "res4b22_relu", "relu6_2", "relu7_2", "relu8_2", "relu9_2"];
    
    % Compute min and max size of anchor boxes based on input image size.
    numAnchorBoxLayers = size(layersToConnect,2);
    [minAnchorSize, maxAnchorSize] = iComputeAnchorBoxeSizes(inputImageSize,numAnchorBoxLayers );
    
    % The following will change based on how anchorBoxLayer is parameterized.
    anchorBoxes = cell(numel(layersToConnect), 1);
    anchorBoxes{1} = iMakeAnchorBoxes(maxAnchorSize(1,1), minAnchorSize(1,1), 2);
    anchorBoxes{2} = iMakeAnchorBoxes(maxAnchorSize(1,2), minAnchorSize(1,2), [3, 2]);
    anchorBoxes{3} = iMakeAnchorBoxes(maxAnchorSize(1,3), minAnchorSize(1,3), [3, 2]);
    anchorBoxes{4} = iMakeAnchorBoxes(maxAnchorSize(1,4), minAnchorSize(1,4), [3, 2]);
    anchorBoxes{5} = iMakeAnchorBoxes(maxAnchorSize(1,5), minAnchorSize(1,5), 2);
    anchorBoxes{6} = iMakeAnchorBoxes(maxAnchorSize(1,6), minAnchorSize(1,6), 2);

    ssdLayerGraph = iCreateSSDLayerGraph(baseLayers, extraLayers, layersToConnect, ...
                         weightsInitializerValue, biasInitializerValue, ...
                         numClasses, ...
                         anchorBoxes);
end
%--------------------------------------------------------------------------
function ssdLayerGraph = createFromCustomNetwork(baseNetwork, inputImageSize, numClasses, predictorBranchNames, anchorBoxes)

    if isa(baseNetwork, 'SeriesNetwork')
        lgraph = layerGraph(baseNetwork.Layers);
    elseif isa(baseNetwork, 'DAGNetwork')
        lgraph = layerGraph(baseNetwork);
    elseif isa(baseNetwork,'nnet.cnn.LayerGraph')
        lgraph = baseNetwork;
    end
        
    lgraph = iReplaceImageInputLayer(lgraph, inputImageSize);

    % Verify that all predictorBranchNames layers exist in lgraph.
    iVerifyLayersExist(lgraph, predictorBranchNames);

    % Remove all layers after the last predictorBranchName.
    lgraph = iRemoveLayers(lgraph, predictorBranchNames(end));

    % Create predictor layers.
    weightsInitializerValue = 'glorot';
    biasInitializerValue = 'zeros';

    ssdLayerGraph = iCreateSSDLayerGraph(lgraph, [], predictorBranchNames, ...
                         weightsInitializerValue, biasInitializerValue, ...
                         numClasses, ...
                         anchorBoxes);
end
%--------------------------------------------------------------------------
function ssdLayerGraph = iCreateSSDLayerGraph(baseLayers, extraLayers, layersToConnect, ...
                         weightsInitializerValue, biasInitializerValue, ...
                         numClasses, ...
                         anchorBoxes)

    % Focal loss parameters
%     alpha = 0.25;
%     gamma = 2;
    alpha = 0.9954;
    gamma = 3.7540;
    % Create layer graph from base network.
    if ~isa(baseLayers, 'nnet.cnn.LayerGraph')
        ssdLayerGraph = layerGraph(baseLayers);
    else
        ssdLayerGraph = baseLayers;
    end

    % Add extra layers if specified.
    if ~isempty(extraLayers)
        lastLayerName = ssdLayerGraph.Layers(end).Name;
        ssdLayerGraph = addLayers(ssdLayerGraph, extraLayers);
        ssdLayerGraph = connectLayers(ssdLayerGraph, lastLayerName, extraLayers(1).Name);
    end

    % Create and add predictor layers, and connect to the right location
    % in the layer graph.
    numClassesPlusBackground = numClasses + 1;
    predictorLayerStruct = iCreatePredictorLayers(...
        layersToConnect, ...
        weightsInitializerValue, biasInitializerValue, ...
        numClassesPlusBackground, ...
        anchorBoxes);
    numFeaturePredictors = numel(predictorLayerStruct);

    mLayer = ssdMergeLayer(numClassesPlusBackground, numFeaturePredictors, 'Name', 'confmerge');
    ssdLayerGraph = addLayers(ssdLayerGraph, mLayer);

    mLayer = ssdMergeLayer(4, numFeaturePredictors, 'Name', 'locmerge');
    ssdLayerGraph = addLayers(ssdLayerGraph, mLayer);

    for idx = 1:numFeaturePredictors
        numBranches = size(predictorLayerStruct(idx).Layers, 1);
        for lidx = 1:numBranches
            ssdLayerGraph = addLayers(ssdLayerGraph, predictorLayerStruct(idx).Layers(lidx));
        end
        %{
        if strcmp(predictorLayerStruct(idx).ConnectTo, "conv4_3")
            l2normlayer = l2NormalizationLayer(20, 'Name', "l2norm");
            ssdLayerGraph = addLayers(ssdLayerGraph, l2normlayer);
            ssdLayerGraph = connectLayers(ssdLayerGraph, "conv4_3", ...
                                               "l2norm");
            predictorLayerStruct(idx).ConnectTo = "l2norm";
        end
        %}
        ssdLayerGraph = connectLayers(ssdLayerGraph, predictorLayerStruct(idx).ConnectTo, ...
                                           predictorLayerStruct(idx).Layers(1).Name);
        ssdLayerGraph = connectLayers(ssdLayerGraph, predictorLayerStruct(idx).Layers(1).Name, ...
                                           predictorLayerStruct(idx).Layers(2).Name);
        ssdLayerGraph = connectLayers(ssdLayerGraph, predictorLayerStruct(idx).Layers(1).Name, ...
                                           predictorLayerStruct(idx).Layers(3).Name);
        ssdLayerGraph = connectLayers(ssdLayerGraph, predictorLayerStruct(idx).Layers(2).Name, "confmerge/in" + num2str(idx));
        ssdLayerGraph = connectLayers(ssdLayerGraph, predictorLayerStruct(idx).Layers(3).Name, "locmerge/in" + num2str(idx));
    end

    classHead = iCreateClassificationHead(alpha, gamma);
    ssdLayerGraph = iConnectHead("confmerge", ssdLayerGraph, classHead);

    regressHead = iCreateRegressionHead();
    ssdLayerGraph = iConnectHead("locmerge", ssdLayerGraph, regressHead);
end

%--------------------------------------------------------------------------
function lgraph = iReplaceImageInputLayer(lgraph,imageSize)
% Replace input size in image input layer.
    idx = arrayfun(@(x)isa(x,'nnet.cnn.layer.ImageInputLayer'),...
        lgraph.Layers);
    imageInputIdx = find(idx,1,'first');
    imageInput =  vision.internal.cnn.utils.updateImageLayerInputSize(...
        lgraph.Layers(imageInputIdx), imageSize);
    lgraph = replaceLayer(lgraph,lgraph.Layers(imageInputIdx).Name,...
        imageInput);
end

%--------------------------------------------------------------------------
function iVerifyLayersExist(lgraph, layerNames)

    numLayers = numel(lgraph.Layers);
    for idx = 1:numel(layerNames)
        foundLayer = false;
        for lIdx = 1:numLayers
            if strcmp(layerNames(idx), lgraph.Layers(lIdx).Name)
                foundLayer = true;
                break;
            end
        end
        if ~foundLayer
            error(message('vision:ssd:InvalidLayerName', layerNames(idx)));
        end
    end
end
%--------------------------------------------------------------------------
function lgraph = iRemoveLayers(lgraph, lastLayer)

    % Remove all the layers after lastLayer.
    dg = vision.internal.cnn.RCNNLayers.digraph(lgraph);

    % Find the last layer.
    id = findnode(dg,char(lastLayer));

    % Search for all nodes starting from the feature extraction
    % layer.
    if ~(sum(id)==0)
        ids = dfsearch(dg,id);
        names = dg.Nodes.Name(ids,:);
        lgraph = removeLayers(lgraph, names(2:end));
    end
end
%--------------------------------------------------------------------------
function layers = iUpdateLastLayersVGG16(layers, weightsInitializerValue, biasInitializerValue)

    % Remove layers that aren't necessary from the vgg16 network. Note that
    % this is specific to vgg16, and may not work for other base networks.

    % Remove all layers of specific types.
    layers(iFindDropoutLayer(layers)) = [];
    layers(iFindSoftmaxLayer(layers)) = [];
    layers(iFindClassificationOutputLayer(layers)) = [];

    % Update all the pool layers to have same padding, instead of 0.
    poolIdx = iFindLayerByName(layers, 'pool', 'partial');
    for idx = poolIdx
        poolLayer = layers(idx);
        newLayer = iUpdateMaxPoolingLayer(poolLayer, 'Stride', poolLayer.Stride, 'Padding', 'same');
        layers(idx) = newLayer;
    end

    % Make pool5 layer have stride [1 1] instead of [2 2].
    pool5Idx = iFindLayerByName(layers, 'pool5', 'exact');
    pool5Layer = layers(pool5Idx);
    newLayer = iUpdateMaxPoolingLayer(pool5Layer, 'Stride', [1 1], 'Padding', 'same');
    layers(pool5Idx) = newLayer;

    % Update fc7 adn fc8 to be Conv2D layers from fully conn.
    idx = iFindLayerByName(layers, 'fc6', 'exact');
    filterSize = 3;
    numFilters = 1024;
    numChannels = 512;
    dilationFactor = 6;
    layers(idx) = convolution2dLayer(filterSize, numFilters, 'NumChannels', numChannels, ...
        'Padding', 'same', ...
        'DilationFactor', dilationFactor, ...
        'Name', 'fc6', ...
        'WeightsInitializer', weightsInitializerValue, ...
        'BiasInitializer', biasInitializerValue);

    idx = iFindLayerByName(layers, 'fc7', 'exact');
    filterSize = 1;
    numFilters = 1024;
    numChannels = 1024;
    layers(idx) = convolution2dLayer(filterSize, numFilters, 'NumChannels', numChannels, ...
        'Padding', 'same', ...
        'Name', 'fc7', ...
        'WeightsInitializer', weightsInitializerValue, ...
        'BiasInitializer', biasInitializerValue);

    % Remove fc8 layer.
    idx = iFindLayerByName(layers, 'fc8', 'exact');
    layers(idx) = [];
end
%--------------------------------------------------------------------------
function extraLayers = iCreateExtraLayers(weightsInitializerValue, biasInitializerValue)

    % Append Extra layers on top of a base network.
    extraLayers = [];

    % Add conv6_1 and corresponding reLU
    filterSize = 1;
    numFilters = 256;
    numChannels = 1024;
    conv6_1 = convolution2dLayer(filterSize, numFilters, 'NumChannels', numChannels, ...
        ... %'Padding', iSamePadding(filterSize),  ...
        'Name', 'conv6_1', ...
        'WeightsInitializer', weightsInitializerValue, ...
        'BiasInitializer', biasInitializerValue);
    relu6_1 = reluLayer('Name', 'relu6_1');
    extraLayers = [extraLayers; conv6_1; relu6_1];

    % Add conv6_2 and corresponding reLU
    filterSize = 3;
    numFilters = 512;
    numChannels = 256;
    conv6_2 = convolution2dLayer(filterSize, numFilters, 'NumChannels', numChannels, ...
        'Padding', iSamePadding(filterSize), ...
        'Stride', [2, 2], ...
        'Name', 'conv6_2', ...
        'WeightsInitializer', weightsInitializerValue, ...
        'BiasInitializer', biasInitializerValue);
    relu6_2 = reluLayer('Name', 'relu6_2');
    extraLayers = [extraLayers; conv6_2; relu6_2];

    % Add conv7_1 and corresponding reLU
    filterSize = 1;
    numFilters = 128;
    numChannels = 512;
    conv7_1 = convolution2dLayer(filterSize, numFilters, 'NumChannels', numChannels, ...
        ... %'Padding', iSamePadding(filterSize),  ...
        'Name', 'conv7_1', ...
        'WeightsInitializer', weightsInitializerValue, ...
        'BiasInitializer', biasInitializerValue);
    relu7_1 = reluLayer('Name', 'relu7_1');
    extraLayers = [extraLayers; conv7_1; relu7_1];

    % Add conv7_2 and corresponding reLU
    filterSize = 3;
    numFilters = 256;
    numChannels = 128;
    conv7_2 = convolution2dLayer(filterSize, numFilters, 'NumChannels', numChannels, ...
        'Padding', iSamePadding(filterSize), ...
        'Stride', [2, 2], ...
        'Name', 'conv7_2', ...
        'WeightsInitializer', weightsInitializerValue, ...
        'BiasInitializer', biasInitializerValue);
    relu7_2 = reluLayer('Name', 'relu7_2');
    extraLayers = [extraLayers; conv7_2; relu7_2];

    % Add conv8_1 and corresponding reLU
    filterSize = 1;
    numFilters = 128;
    numChannels = 256;
    conv8_1 = convolution2dLayer(filterSize, numFilters, 'NumChannels', numChannels, ...
        ... %'Padding', iSamePadding(filterSize),  ...
        'Name', 'conv8_1', ...
        'WeightsInitializer', weightsInitializerValue, ...
        'BiasInitializer', biasInitializerValue);
    relu8_1 = reluLayer('Name', 'relu8_1');
    extraLayers = [extraLayers; conv8_1; relu8_1];

    % Add conv8_2 and corresponding reLU
    filterSize = 3;
    numFilters = 256;
    numChannels = 128;
    conv8_2 = convolution2dLayer(filterSize, numFilters, 'NumChannels', numChannels, ...
        ... %'Padding', iSamePadding(filterSize),  ...
        ... %'Stride', [2, 2], ... % is this necessary?
        'Name', 'conv8_2', ...
        'WeightsInitializer', weightsInitializerValue, ...
        'BiasInitializer', biasInitializerValue);
    relu8_2 = reluLayer('Name', 'relu8_2');
    extraLayers = [extraLayers; conv8_2; relu8_2];

    % Add conv9_1 and corresponding reLU
    filterSize = 1;
    numFilters = 128;
    numChannels = 256;
    conv9_1 = convolution2dLayer(filterSize, numFilters, 'NumChannels', numChannels, ...
        'Padding', iSamePadding(filterSize), ...
        'Name', 'conv9_1', ...
        'WeightsInitializer', weightsInitializerValue, ...
        'BiasInitializer', biasInitializerValue);
    relu9_1 = reluLayer('Name', 'relu9_1');
    extraLayers = [extraLayers; conv9_1; relu9_1];

    % Add conv9_2 and corresponding reLU
    filterSize = 3;
    numFilters = 256;
    numChannels = 128;
    conv9_2 = convolution2dLayer(filterSize, numFilters, 'NumChannels', numChannels, ...
        'Name', 'conv9_2', ...
        'WeightsInitializer', weightsInitializerValue, ...
        'BiasInitializer', biasInitializerValue);
    relu9_2 = reluLayer('Name', 'relu9_2');
    extraLayers = [extraLayers; conv9_2; relu9_2];
end
%--------------------------------------------------------------------------
function predictorLayerStruct = iCreatePredictorLayers(...
    layersToConnect, weightsInitializerValue, biasInitializerValue, ...
    numClasses, anchorBoxes)

    % Create independent predictor layers for each branch.
    numPredictorBranches = numel(layersToConnect);
    predictorLayerStruct = repmat(struct('ConnectTo', '', 'Layers', []), ...
                                  numPredictorBranches, 1);

    for idx = 1:numPredictorBranches
        pbLayer = anchorBoxLayer(anchorBoxes{idx}, 'Name', layersToConnect(idx) + "_anchorbox");
        
        conv_layers = iCreateConvHead(layersToConnect(idx), ...
                          weightsInitializerValue, biasInitializerValue, ...
                          numClasses, ...
                          size(anchorBoxes{idx}, 1));
        layerList = [pbLayer; conv_layers];
        predictorLayerStruct(idx).ConnectTo = layersToConnect(idx);
        predictorLayerStruct(idx).Layers = layerList;
    end
end
%--------------------------------------------------------------------------
function newLayer = iUpdateMaxPoolingLayer(oldLayer, varargin)
% Extract padding value from the old layer.
padVal = iPadValFromMode(oldLayer);

p = inputParser;
p.addParameter('Stride',oldLayer.Stride);
p.addParameter('Padding',padVal);

p.parse(varargin{:});

newLayer = maxPooling2dLayer(oldLayer.PoolSize,...
    'Stride',p.Results.Stride,...
    'Name',oldLayer.Name,...
    'Padding',p.Results.Padding);

end
%--------------------------------------------------------------------------
function padVal = iPadValFromMode(layer)
% Extract padding value from the old layer.
if strcmp(layer.PaddingMode,'manual')
    padVal = layer.PaddingSize;
else
    padVal = layer.PaddingMode;
end
end

%--------------------------------------------------------------------------
function boxes = iMakeAnchorBoxes(boxHeight, boxWidth, aspectRatio)

    boxes = [boxHeight boxWidth];
    boxes = [boxes; boxWidth boxHeight];

    for arIdx = 1:numel(aspectRatio)
        ar = aspectRatio(arIdx);
        boxes = [boxes; boxHeight boxWidth/sqrt(ar)]; %#ok<AGROW>
        boxes = [boxes; boxHeight/sqrt(ar) boxWidth]; %#ok<AGROW>
    end

    boxes = round(boxes);
end

%--------------------------------------------------------------------------
function [minAnchorSize,maxAnchorSize] = iComputeAnchorBoxeSizes(inputImageSize, numAnchorBoxLayers)
    
    %Liu, Wei, et al. "Ssd: Single shot multibox detector." European
    %conference on computer vision. Springer, Cham, 2016.
    minRatio = 20;
    maxRatio = 90;
    % 2 is subtracted to match authors implementation. See Line 310 in
    % https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_pascal.py
    step = floor((maxRatio-minRatio)./(numAnchorBoxLayers-2));
    idx = 1;
    % AnchorBox sizes are computed based on minimum dimension of input
    % image.
    minDim = min(inputImageSize(1,1),inputImageSize(1,2));
    for i = minRatio:step:maxRatio
        minAnchorSize(idx) = (minDim.* i)./100; %#ok<AGROW>
        maxAnchorSize(idx) = (minDim.* (i + step))./100; %#ok<AGROW>
        idx = idx+1;
    end
    minAnchorSize = [(minDim *10)./100, minAnchorSize];
    maxAnchorSize = [(minDim *20)./100, maxAnchorSize];
end


%--------------------------------------------------------------------------
function p = iSamePadding(FilterSize)
    p = floor(FilterSize / 2);
end
%--------------------------------------------------------------------------
function layers = iCreateConvHead(layerName, weightsInitializerValue, biasInitializerValue, numClasses, numBoxesPerGrid)

    layers = [];
    % Add mbox_conf
    filterSize = 3;
    numFilters = numClasses * numBoxesPerGrid;
    layer_mbox_conf = convolution2dLayer(filterSize, numFilters, ...
        'Padding', iSamePadding(filterSize), ...
        'Name', layerName + "_mbox_conf", ...
        'WeightsInitializer', weightsInitializerValue, ...
        'BiasInitializer', biasInitializerValue);
    layers = [layers; layer_mbox_conf];

    % Add mbox_loc
    numBBoxElems = 4;
    filterSize = 3;
    numFilters = numBBoxElems * numBoxesPerGrid;
    layer_mbox_loc = convolution2dLayer(filterSize, numFilters, ...
        'Padding', iSamePadding(filterSize), ...
        'Name', layerName + "_mbox_loc", ...
        'WeightsInitializer', weightsInitializerValue, ...
        'BiasInitializer', biasInitializerValue);
    layers = [layers; layer_mbox_loc];
end
%--------------------------------------------------------------------------
function lgraph = iCreateRegressionHead()

    lgraph = layerGraph();
    regressLayerName = "anchorBoxRegression";
    regressionLayer = rcnnBoxRegressionLayer('Name',regressLayerName);
    lgraph = addLayers(lgraph, regressionLayer);
end
%--------------------------------------------------------------------------
function lgraph = iCreateClassificationHead(alpha, gamma)

    lgraph = layerGraph();
    softmaxLayerName = "anchorBoxSoftmax";
    abSoftmaxLayer = softmaxLayer('Name', softmaxLayerName);
    lgraph = addLayers(lgraph, abSoftmaxLayer);

    focallossLayerName = "focalLoss";
    classificationOutputLayer = focalLossLayer(gamma, alpha, 'Name', focallossLayerName);
    lgraph = addLayers(lgraph, classificationOutputLayer);
    lgraph = connectLayers(lgraph, softmaxLayerName, focallossLayerName);
end
%--------------------------------------------------------------------------
function lgraph = iConnectHead(layerName, lgraph, head)

    lgraph = addLayers(lgraph, head.Layers);
    lgraph = connectLayers(lgraph, layerName, head.Layers(1).Name);
end
%--------------------------------------------------------------------------
function idx = iFindDropoutLayer(layers)

    idx = iFindLayer(layers, 'nnet.cnn.layer.DropoutLayer');
end
%--------------------------------------------------------------------------
function idx = iFindSoftmaxLayer(layers)

    idx = iFindLayer(layers, 'nnet.cnn.layer.SoftmaxLayer');
end
%--------------------------------------------------------------------------
function idx = iFindClassificationOutputLayer(layers)

    idx = iFindLayer(layers, 'nnet.cnn.layer.ClassificationOutputLayer');
end
%--------------------------------------------------------------------------
function idx = iFindLayer(layers, type)

    results = arrayfun(@(x)isa(x,type),layers,'UniformOutput', true);
    idx = find(results);
end
%--------------------------------------------------------------------------
function idx = iFindLayerByName(layers, layerName, matchType)

    if strcmp(matchType, 'exact')
        results = cellfun(@(x)strcmp(x, layerName),{layers.Name},'UniformOutput', true); 
    elseif strcmp(matchType, 'partial')
        results = cellfun(@(x)contains(x, layerName),{layers.Name},'UniformOutput', true); 
    end
    idx = find(results);
end
%--------------------------------------------------------------------------
function [baseNetwork, imageSize, numClasses] = iParseInputs(imageSize, numClasses, baseNetwork)

    if isstring(baseNetwork) || ischar(baseNetwork)
        % Construct using pre-defined network names.
        supportedNetworks = ["vgg16", "resnet101", "resnet50"];
        baseNetwork = validatestring(baseNetwork, supportedNetworks, mfilename, 'baseNetwork', 1);
    else
        validateattributes(baseNetwork, {'nnet.cnn.LayerGraph', 'SeriesNetwork', 'DAGNetwork', 'string', 'char'}, ...
            {'scalar'}, mfilename, ...
            'baseNetwork', 2);
    end

    validateattributes(imageSize, {'numeric'}, ...
        {'size', [1, 3], 'nonempty', 'positive', 'integer'}, mfilename,...
        'inputImageSize', 2);
    validateattributes(numClasses, {'numeric'}, ...
        {'scalar','positive','integer','real','finite','nonempty','nonsparse'}, mfilename,...
        'numClasses', 3);
end
%--------------------------------------------------------------------------
function [predictorLayerNames, anchorBoxes] = iParseOptionalInputs(varargin)

    % Validate optional inputs for custom network creation.
    narginchk(2,2);
    parser = inputParser();
    inputNumber = 4;

    % validate anchorBoxes
    validateAnchorBoxes = @(x)validateattributes(x, {'cell'}, {'row', 'size', [1 NaN]}, ...
        mfilename, 'AnchorBoxes', inputNumber);
    inputNumber = inputNumber + 1;
    parser.addRequired('AnchorBoxes', validateAnchorBoxes);

    % validate predictorBranchNames
    validateLayerNames = @(x)validateattributes(x, {'cell','string'}, {'row', 'size', [1 NaN]}, ...
        mfilename, 'predictorLayerNames', inputNumber);
    parser.addRequired('predictorLayerNames', validateLayerNames);
    
    parser.parse(varargin{:});
    params = parser.Results;

    anchorBoxes = params.AnchorBoxes;
    for idx = 1:numel(anchorBoxes)
        validateattributes(anchorBoxes{idx}, {'numeric'}, {'size', [NaN, 2], 'positive', 'integer'}, ...
        mfilename, 'anchorBoxes');
    end
    
    predictorLayerNames = params.predictorLayerNames;
    if ~(iscellstr(predictorLayerNames) || isstring(predictorLayerNames))
        error(message('vision:ssd:InvalidPredictorLayerNames'));
    end
    
    if ~isequal(size(anchorBoxes,2), numel(predictorLayerNames))
        error(message('vision:ssd:InvalidAnchorBoxes'));
    end
    
    predictorLayerNames = string(predictorLayerNames);
        
end
