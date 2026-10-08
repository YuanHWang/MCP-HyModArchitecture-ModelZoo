clc;clear;
z=importdata('LeafRiverDaily.txt');
zz=cat(2,z(:,3),linspace(1,size(z,1),size(z,1))');
%%
Qz=sortrows(zz,1);pairno = size(z,1)/2;
PQz.Train=[];
PQz.Select=[];
PQz.Test=[];
for iloc=1:pairno
    temp=[];
    temp=cat(1,temp,Qz(iloc,:),Qz(size(z,1)-iloc+1,:));
    flag=mod(iloc,4);
    if(flag<=1)
       temp=cat(2,temp,linspace(-1,-1,2)');
       PQz.Train=cat(1,PQz.Train,temp);
    elseif(flag==2)
       temp=cat(2,temp,linspace(1,1,2)');        
        PQz.Test=cat(1,PQz.Test,temp);
    elseif(flag==3)
       temp=cat(2,temp,linspace(0,0,2)');        
        PQz.Select=cat(1,PQz.Select,temp);
    end
end
%%
histogram(PQz.Train(:,1),'normalization','pdf')
hold on
histogram(PQz.Test(:,1),'normalization','pdf')
hold on
histogram(PQz.Select(:,1),'normalization','pdf')
xlim([0 max(z(:,3))])
% set(gca,'XScale','log');
close
%%
histogram(z(1:7305,3),'normalization','pdf')
hold on
histogram(z(7306:14610,3),'normalization','pdf')
hold on
xlim([0 max(z(:,3))])
close
%%
PQz.final=cat(1,PQz.Train,PQz.Select,PQz.Test);
PQz.final=sortrows(PQz.final,2);
%%
PQz.spinup=cat(1,PQz.final(1:365,:),PQz.final(1:365,:),PQz.final(1:365,:));
PQz.spinup(:,3)=-99999;
PQz.final=cat(1,PQz.spinup,PQz.final);
csvwrite('LeafRiverDaily_43YR_Flag.txt',PQz.final(:,3));
csvwrite('LeafRiverDaily_23YR_Flag.txt',PQz.final(1:8400,3));
z43=cat(1,z(1:365,:),z(1:365,:),z(1:365,:),z);