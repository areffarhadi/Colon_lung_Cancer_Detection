% Transfer Learning Using Squeezenet
 close all
 clear

% images = imageDatastore('.\alldata',...
%     'IncludeSubfolders',true,...
%     'LabelSource','foldernames');
%[trainingImages,validationImages] = splitEachLabel(images,0.75,'randomized');
%%
% Divide the data into training and validation data sets. Use 75% of the

 load('trainvaliddata.mat')

numTrainImages = numel(trainingImages.Labels);

%% Load Pretrained Network

 load('squeezenet.mat');

%% Transfer Layers to New Network

lgraph = layerGraph(net);
numClasses = numel(categories(trainingImages.Labels));

newConvLayer =  convolution2dLayer([1, 1],numClasses,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10,"Name",'new_conv');
lgraph = replaceLayer(lgraph,'conv10',newConvLayer);
newClassificatonLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassificatonLayer);
 %%%
%%

miniBatchSize = 32;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4,...
    'Plots','training-progress',...
    'Verbose',false,...
    'ValidationData',validationImages,...
    'Shuffle', 'every-epoch', ...
    'ValidationFrequency',numIterationsPerEpoch);
    
%%
% Train the network that consists of the transferred and new layers.

netTransfer = trainNetwork(trainingImages,lgraph,options);

save('netTransfer','netTransfer');
%% Classify Validation Images
% Classify the validation images using the fine-tuned network.
predictedLabels = classify(netTransfer,validationImages);

%%
% Calculate the classification accuracy on the validation set. Accuracy is
% the fraction of labels that the network predicts correctly.
valLabels = validationImages.Labels;
accuracy = mean(predictedLabels == valLabels);

