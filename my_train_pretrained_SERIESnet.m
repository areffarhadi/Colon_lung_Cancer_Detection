% Transfer Learning Using AlexNet

% close all
% clear

load('trainvaliddata.mat')
%%
% Divide the data into training and validation data sets. Use 75% of the

%    [trainingImages,validationImages] = splitEachLabel(images,0.75,'randomized');

%%
% This very small data set now contains 55 training images and 20
% validation images. Display some sample images.
numTrainImages = numel(trainingImages.Labels);

%% Load Pretrained Network

         load('alexnet.mat');
         net=alexnet;

%% Transfer Layers to New Network

layersTransfer = net.Layers(1:end-3);
 numClasses = numel(categories(trainingImages.Labels));
 layers = [
     layersTransfer
     fullyConnectedLayer(numClasses,'WeightLearnRateFactor',1,'BiasLearnRateFactor',2)
     softmaxLayer
     classificationLayer];
 
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

 netTransfer = trainNetwork(trainingImages,layers,options);

save('netTransfer','netTransfer');
%% Classify Validation Images
% Classify the validation images using the fine-tuned network.
predictedLabels = classify(netTransfer,validationImages);

%%
% Calculate the classification accuracy on the validation set. Accuracy is
% the fraction of labels that the network predicts correctly.
valLabels = validationImages.Labels;
accuracy = mean(predictedLabels == valLabels)

