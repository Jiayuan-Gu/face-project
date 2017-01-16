function [imgInfo] = preprocess_dataset()
%PREPROCESS_DATASET Summary of this function goes here
%   Detailed explanation goes here
    
    clear;
    clc;
    delete(gcp('nocreate'));
    diary([datestr(now,13) '.log'])
    diary on;
    
    numCPU = 16;
    parpool(numCPU);
    
    datasetDir = '/data-disk/gujy/Face/dataset/WF';
    listFile = '/data-disk/gujy/Face/dataset/WF/v1/wider_face_train.mat';
    imgDir = [datasetDir '/WIDER_train/images'];
    outputDir = '/data-disk/gujy/Face/WF/preprocess';
    
    load(listFile);
    
    if ~exist(outputDir,'dir')
        mkdir(outputDir);
    end
    
    nEvent = length(event_list);
    imgInfo = [];
    
    for iEvent = 1:nEvent
%         eventInd = cell2mat(regexp(event_list{iEvent},'\d', 'match'));
        eventDir = [outputDir '/' event_list{iEvent}];
        if ~exist(eventDir,'dir')
            mkdir(eventDir);
        end
        fileList = file_list{iEvent};
        bbxList = face_bbx_list{iEvent};
        invalidList = invalid_label_list{iEvent};
        
        nFile = length(fileList);
        imgSize = cell(nFile,1);
        parfor iFile = 1:nFile
            [~,imgName,~] = fileparts(fileList{iFile});
            fprintf('%s:%s.\n',datestr(now,13),imgName);
            try
                img = imread([imgDir '/' event_list{iEvent} '/' fileList{iFile} '.jpg']);
            catch
                disp('read img error.');
                continue;
            end
            imgSize{iFile} = size(img);
            label = zeros(size(img,1),size(img,2));
            
            bbxs = bbxList{iFile};
            bbxs = bbxs(~invalidList{iFile},:);
            nBox = size(bbxs,1);
            for iBox = 1:nBox
                bbx = bbxs(iBox,:);
                x1 = max(1,floor(bbx(1)));
                x2 = min(size(img,2),ceil(bbx(1)+bbx(3)));
                y1 = max(1,floor(bbx(2)));
                y2 = min(size(img,1),ceil(bbx(2)+bbx(4)));
                label(y1:y2,x1:x2) = iBox;
            end
            outputName = [eventDir '/' imgName];
            save_data(img,label,bbxs,imgName,outputName);
        end
        imgInfo = [imgInfo;imgSize];
    end
    diary off;
end

function save_data(img,label,bbxs,imgName,outputName)
    save([outputName '.mat'],'img','label','bbxs','imgName');
%     imwrite(img,[outputName '.jpg'])
%     imwrite(label,[outputName '_bbx.jpg'])
end

