images = imageDatastore('.\alldata',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

Des_size=[256,256];

for i=1:25000
    m1=imread(images.Files{i, 1});
    m1=imresize(m1,Des_size);
    imwrite(m1,images.Files{i, 1})
    disp(i)

end