% test pretrained Squeezenet in one of the Folds(K=4)

%  close all
%  clear
 
train = false;

  load('trainvaliddata.mat')
  classes={'colon_aca';'colon_n';'lung_aca';'lung_n';'lung_scc'};

%% Load Pretrained Network
 net=squeezenet();


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
    'MaxEpochs',15,...
    'InitialLearnRate',1e-4,...
    'Plots','training-progress',...
    'Verbose',false,...
    'ValidationData',validationImages,...
    'Shuffle', 'every-epoch', ...
    'ValidationFrequency',numIterationsPerEpoch);
%    
%%
% Train the network that consists of the transferred and new layers.
if train == true
    netTransfer = trainNetwork(trainingImages,lgraph,options);

    save('netTransfer','netTransfer');
    predictedLabels = classify(netTransfer,validationImages);
else  
    load('net_from_Squeezenet.mat');
    predictedLabels = classify(netTransfer,validationImages);
    yp = predict(netTransfer,validationImages);
end


%%
% Calculate the classification accuracy on the validation set. 
valLabels = validationImages.Labels;
accuracy = mean(predictedLabels == valLabels);
disp(accuracy);

cm=confusionchart(validationImages.Labels,predictedLabels);
figure()
plotconfusion(validationImages.Labels,predictedLabels);

[C,order] = confusionmat(validationImages.Labels,predictedLabels);
stats = statsOfMeasure(C, 1);

figure()
rocObj = rocmetrics(validationImages.Labels,yp,classes);
plot(rocObj,AverageROCType="micro")

