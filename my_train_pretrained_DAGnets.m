% Transfer Learning Using AlexNet

% close all
% clear
% images = imageDatastore('QazvinData',...
%     'IncludeSubfolders',true,...
%     'LabelSource','foldernames');
% PedDatasetPath = ('D:\UA Speech\images'); 
%  cate ={'D1','D2','D3','D4','D5','D6','D7','D8','D9','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','LD','LA'};
% cate ={'D1','D2','D3','D4','D5','D6','D7','D8','D9','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16'};

% images = imageDatastore(fullfile(PedDatasetPath, cate), 'LabelSource', 'foldernames');
% images = imageDatastore('G:\alldata',...
%     'IncludeSubfolders',true,...
%     'LabelSource','foldernames');
% trainingImages = imageDatastore('G:\MY_Code\train',...
%     'IncludeSubfolders',true,...
%     'LabelSource','foldernames');
% validationImages = imageDatastore('G:\MY_Code\test',...
%     'IncludeSubfolders',true,...
%     'LabelSource','foldernames');
load('trainvaliddata.mat')
%%
% Divide the data into training and validation data sets. Use 70% of the
% images for training and 30% for validation. |splitEachLabel| splits the
% |images| datastore into two new datastores.


%    [trainingImages,validationImages] = splitEachLabel(images,0.75,'randomized');
%  load('30WordValandTrainData_80-20.mat');
% load('30WordValandTrainData.mat');
% trainingImages=imdsTrain;
% validationImages=imdsValidation;
%%
% This very small data set now contains 55 training images and 20
% validation images. Display some sample images.
numTrainImages = numel(trainingImages.Labels);
% idx = randperm(numTrainImages,16);
% figure
% for i = 1:16
%     subplot(4,4,i)
%     I = readimage(trainingImages,idx(i));
%     imshow(I)
% end

%% Load Pretrained Network

%     net = nasnetlarge();
%     net = densenet201();
%      load('efficientnetb0.mat');
   load('Squeezenet.mat');
%   net=netTransfer;
% net=inceptionresnetv2();

%%
% Display the network architecture. The network has five convolutional
% layers and three fully connected layers.

% net.Layers

% savedir='D:\saveNet';

%% Transfer Layers to New Network

 lgraph = layerGraph(net);
numClasses = numel(categories(trainingImages.Labels));

newLearnableLayer = fullyConnectedLayer(numClasses, ...
'Name','new_fc', ...
'WeightLearnRateFactor',10, ...
'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'predictions',newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);
 %%%
%%

miniBatchSize = 16;
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
%    
%[varargout{1:aLayer.NumOutputs+aLayer.PrivateNumStates}] = predict( aLayer, varargin{:} );
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
accuracy = mean(predictedLabels == valLabels)

