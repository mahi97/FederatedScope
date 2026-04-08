#!/bin/sh


quality_list="
1
2
3
4
5
"


data1=DIV2K
data2=Flickr30k
data4=Flickr30koverlap2
loss_type1=PSNR
loss_type2=SSIM
# ours
#qsub -N MambaIC_DIV2K -v model_name=MambaIC,data_info=${data1},loss_type=${loss_type} qsub.sh_train_PSNR

#qsub -N GMMIC_AR_DIV2K -v model_name=GMMIC_AR,data_info=${data1},loss_type=${loss_type} qsub.sh_train_PSNR

#qsub -N MG_GMMIC_AR_DIV2K -v model_name=GMMIC_AR,data_info=${data1},loss_type=${loss_type} qsub.sh_train_MG_PSNR

#qsub -N MG_GMMIC_AR_DIV2K -v model_name=GMMIC_AR,data_info=${data2},loss_type=${loss_type} qsub.sh_train_MG_PSNR

#qsub -N WACNN_DIV2K -v model_name=WACNN,data_info=${data1},loss_type=${loss_type1} qsub.sh_train_MG_PSNR

#qsub -N WACNN_Flick -v model_name=WACNN,data_info=${data2},loss_type=${loss_type1} qsub.sh_train_MG_PSNR

#qsub -N STF_DIV2K -v model_name=STF,data_info=${data1},loss_type=${loss_type1} qsub.sh_train_MG_PSNR

#qsub -N STF_Flick -v model_name=STF,data_info=${data2},loss_type=${loss_type1} qsub.sh_train_MG_PSNR

#qsub -N TCM_DIV2K -v model_name=TCM,data_info=${data1},loss_type=${loss_type1} qsub.sh_train_MG_PSNR

#qsub -N TCM_Flick -v model_name=TCM,data_info=${data2},loss_type=${loss_type1} qsub.sh_train_MG_PSNR

#qsub -N GMMAtt_DIV2K -v model_name=GMMAtt,data_info=${data1},loss_type=${loss_type1} qsub.sh_train_MG_PSNR

#qsub -N GMMAtt_Flick -v model_name=GMMAtt,data_info=${data2},loss_type=${loss_type1} qsub.sh_train_MG_PSNR

#qsub -N HPCM_DIV2K -v model_name=HPCM,data_info=${data1},loss_type=${loss_type1} qsub.sh_train_MG_PSNR

#qsub -N HPCM_Flick -v model_name=HPCM,data_info=${data2},loss_type=${loss_type1} qsub.sh_train_MG_PSNR

#qsub -N WACNN1 -v model_name=WACNN1,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR
#qsub -N WACNN2 -v model_name=WACNN2,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR
#qsub -N WACNN3 -v model_name=WACNN3,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR
#qsub -N WACNN4 -v model_name=WACNN4,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR
#qsub -N WACNN5 -v model_name=WACNN5,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR


#qsub -N HPCM1 -v model_name=HPCM1,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR
#qsub -N HPCM2 -v model_name=HPCM2,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR
#qsub -N HPCM3 -v model_name=HPCM3,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR
#qsub -N HPCM4 -v model_name=HPCM4,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR
#qsub -N HPCM5 -v model_name=HPCM5,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR


# qsub -N SASIC -v model_name=SASIC,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR
# qsub -N ECSIC -v model_name=ECSIC,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR
#qsub -N BiSIC -v model_name=BiSIC,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR
#qsub -N BiSICFast -v model_name=BiSICFast,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR
#qsub -N CAMIC -v model_name=CAMIC,data_info=${data4},loss_type=${loss_type1} qsub.sh_train_MG_PSNR


qsub -N MAHI_AP  qsub.sh_mahi_aprils