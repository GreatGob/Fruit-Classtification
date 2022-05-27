clc;
close all;
clear all;
format long
hur = zeros(2,7);
scrFile = dir('E:\7\Pattern Recognition\Matlab\Hu_KNN\Apple_binary\*.png');
for i = 1: length(scrFile)
    filename = strcat('E:\7\Pattern Recognition\Matlab\Hu_KNN\Apple_binary\',scrFile(i).name);
    S{i} = imread(filename);  
    %S = rgb2gray(I);
    %S{i} = 255 - S{i};
    S{i} = S{i}/255;  
    [S1, S2, S3, S4, S5, S6, S7] = Function_Hu(S{i});
    
    hur(i,:)= [S1, S2, S3, S4, S5, S6, S7];
  hu_moments_vector_norm(i,:) = -sign(hur(i,:)).*(log10(abs(hur(i,:))))
 
end

 xlswrite('apple.xlsx', hu_moments_vector_norm);