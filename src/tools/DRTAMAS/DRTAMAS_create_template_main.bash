#!/bin/bash


my_RUN () 
{
  cmd="$1"
  echo "Running: ${cmd}" 
  ${cmd}
  
  if [ ! $? -eq 0 ] 
  then
     echo ""
     echo "Error running command ${cmd}"
     exit
  fi 
}




change_date='12/14/3517'

echo Population average DTI image creation tool with DR-TAMAS. Date ${change_date}.. Yes... DR-TAMAS is a product of the future....We acquired it with time travel... For the details of our time travel technique please refer to www.backtothefuture.com
echo ''


if [ $# -lt 2 ]
then
    echo 'Usage: DRTAMAS_create_template_with_structurals  options'
    echo 'OPTIONS:'
    echo '   -s=full_path_to_textfile_with_tensors  or --subjects=full_path_to_textfile_with_tensors  (required)'
    echo '   -si=full_path_to_textfile_with_structurals  or --subjects_structurals=full_path_to_textfile_with_structurals  (required).Structural images in a given line shouldbe separated by commas.'
    echo '   -ss={0,1,2}  or --start_step={0,1,2} (optional)'
    echo '       0:rigid (default)'
    echo '       1:affine'
    echo '       2:diffeomorphic'
    echo '   -i=full_path_to_initial_rigid_template or --initial_rigid_template=full_path_to_initial_rigid_template (optional)'
    echo '   -c={0,1}   or --constrain_deformations={0,1} (optional. default 1)'
    echo '       1:forces the average of all deformation fields to be zero.'
    echo '   -r=final_resolution   or --resolution=final_resolution (optional. the (isotropic) resolution of all registered images in mm.)'
    echo '   -o=output_folder   or --output_folder=full_path_to_output_folder (optional. if not provided, files are placed in the folder of the textfile of images. )'
    echo '   -ds=diffeomorphic_registration_start_iteration or --diffeomorphic_registration_start_iteration=diffeomorphic_registration_start_iteration (optional. default:0'
    exit
fi

mcurrdir=$(pwd)
MY_PATH="$(dirname -- "${BASH_SOURCE[0]}")"            # relative
MY_PATH="$(cd -- "$MY_PATH" && pwd)"    # absolutized and normalized
if [[ -z "$MY_PATH" ]] ; then
  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
export PATH=${PATH}/:${MY_PATH}
cd ${mcurrdir}

subjects_structurals=''
subjects=''
start_step=0
Niter=6
initial_rigid_template=''
constrain_deformations=1
resolution=0
output_folder=''
diffeomorphic_registration_start_iteration=0




for i in "$@"
do
case $i in
    -s=*|--subjects=*)
        subjects="${i#*=}"
        shift 
    ;;
   -ds=*|--diffeomorphic_registration_start_iteration=*)
        diffeomorphic_registration_start_iteration="${i#*=}"
        shift 
    ;;
    -si=*|--subjects_structurals=*)
        subjects_structurals="${i#*=}"
        shift 
    ;;
    -r=*|--resolution=*)
        resolution="${i#*=}"
        shift 
    ;;
    -ss=*|--start_step=*)
       start_step="${i#*=}"
        shift 
    ;;
    -i=*|--initial_rigid_template=*)
        initial_rigid_template="${i#*=}"
        shift
    ;;
    -c=*|--constrain_deformations=*)
        constrain_deformations="${i#*=}"
        shift
    ;;
    -o=*|--output_folder=*)
        output_folder="${i#*=}"
        shift 
    ;;
    *)
        echo Unrecognized command line option ${i}.  Exiting
        exit
    ;;
    *)
            # unknown option
    ;;
esac
done




if [ ! -e "${subjects}" ]
then
     echo Subject_file: ${subjects} does not exist. Exiting....
     exit
fi


if [ ! -z "${subjects_structurals}" ]
then
    if [ ! -e "${subjects_structurals}" ]
    then
         echo Subject_structurals file: ${subjects_structurals} does not exist. Exiting....
         exit
    fi
fi


if [ !  -z  $output_folder  ]
then
     if [ ! -e "${output_folder}" ]
     then
          mkdir -p  ${output_folder}
     fi
     cp ${subjects} ${output_folder}/
     subjects=${output_folder}/$(basename "${subjects}")
     
     if [ ! -z "${subjects_structurals}" ]
     then
         cp ${subjects_structurals} ${output_folder}/
         subjects_structurals=${output_folder}/$(basename "${subjects_structurals}")
     fi
fi



if [ ${start_step} -ne 0  ] && [ ${start_step} -ne 1  ] && [ ${start_step} -ne 2  ]
then
     echo Start_step: ${start_step} not 0, 1 or 2. Exiting....
     exit
fi




if [ ! -z "$initial_rigid_template" ]
then
    if [ ! -e "${subjects}" ]
    then
        echo Initial rigid template $initial_rigid_template does not exist. Exiting.... 
        exit
    fi
fi



if [ ${constrain_deformations} -ne 0  ] && [ ${constrain_deformations} -ne 1  ] 
then
     echo 'constrain_deformations not 0 or 1. Exiting....'
     exit
fi





echo Subjects:                  ${subjects}
echo Subjects_structurals:      ${subjects_structurals}
echo Start step:                ${start_step}
echo Initial rigid template:    ${initial_rigid_template}
echo Num parallel DTIREG runs:  ${ncpus}
echo Constrain deformations:    ${constrain_deformations}
echo Registration resolution:   ${resolution}



listdir=$(dirname "${subjects}")

if [  -e "${listdir}/command.log" ]
then
    rm ${listdir}/command.log
fi

echo DR_TAMAS vdate: ${change_date} >>${listdir}/command.log
echo Subjects:                  ${subjects}>>${listdir}/command.log
echo Start step:                ${start_step}>>${listdir}/command.log
echo Initial rigid template:    ${initial_rigid_template}>>${listdir}/command.log
echo Constrain deformations:    ${constrain_deformations}>>${listdir}/command.log
echo Registration resolution:   ${resolution}>>${listdir}/command.log

let Nstructurals=0
if [ ! -z "${subjects_structurals}" ]
then
    first_line=$(head -n 1 ${subjects_structurals})
    IFS=','
    read -ra structurals <<< "$first_line"
    let Nstructurals=${#structurals[@]}
    IFS=$' \t\n'
fi

subjects_diffeo=`echo $subjects | sed -e 's/.txt/_diffeo.txt/'`
if [ -e "${subjects_diffeo}" ]
then
        rm -rf ${subjects_diffeo}  
fi


for ((i=0; i<Nstructurals; i++))
do
    subjects_structurals_diffeo=`echo $subjects_structurals | sed -e "s/.txt/_diffeo_${i}.txt/"`
    if [ -e "${subjects_structurals_diffeo}" ]
    then
        rm -rf ${subjects_structurals_diffeo}  
    fi
done


deformation_fields=`echo $subjects | sed -e 's/.txt/_defs.txt/'`
if [ -e "${deformation_fields}" ]
then
        rm -rf ${deformation_fields}  
fi



aa=0
for subj in `cat ${subjects}`
do          
    if [ -e ${subj} ]
    then
        name=${subj}
    else
        name=${listdir}/${subj}
    fi

    subjects_list[$aa]=${name}
    aa=$((aa+1))
done
nsubjects=${#subjects_list[@]}




#######################################################SELECT THE MOST REPRESENTATIVE TEMPLATE ###############################

if [ ${start_step} -le 0 ] 
then
    if [ -z "$initial_rigid_template" ]; then
        echo "Initial rigid template NOT provided..."
        echo "Selecting the most representative template"
        my_RUN "SelectMostRepresentativeSample ${subjects} ${listdir}/average_rigid_0.nii"
    else
        echo "Initial rigid template provided..."
        echo "Copying onto average_rigid_0.nii"

        cp $initial_rigid_template ${listdir}/average_rigid_0.nii
    fi
fi


#######################################################RESAMPLE THE INITIAL TEMPLATE ###############################

if [ ${start_step} -le 0 ] 
then
    if [ $(echo "$resolution != 0" | bc) -eq 1 ]
    then
        in_temp=${listdir}/average_rigid_0.nii
        cp ${in_temp} ${listdir}/average_rigid_0_orig.nii

        ResampleTensor ${listdir}/average_rigid_0_orig.nii ${resolution} ${resolution} ${resolution}        
        mv ${listdir}/average_rigid_0_orig_resampled.nii ${listdir}/average_rigid_0.nii
    fi
fi




#######################################################INITIAL RIGID ALIGNMENT ###############################



count=0
subjects_affine=`echo $subjects | sed -e 's/.txt/_affine.txt/'`  
    
if [ -e "${subjects_affine}" ]
then
        rm -rf ${subjects_affine}  
fi

rigid_list=${listdir}/rigids.txt
if [ -e ${rigid_list} ]    
then 
    rm ${rigid_list}
fi


cnt=0

for subj in `cat ${subjects}`
do
    filename=$(basename "${subj}")
    extension="${filename##*.}"
    filename="${filename%.*}"

	    
    if [ -e "${subj}" ]
    then
        name=${subj}
    else
        name=${listdir}/${subj}
    fi

    fixed_name=${listdir}/average_rigid_0.nii

    
   echo ${filename}_aff.nii >> ${subjects_affine}
   echo ${listdir}/${filename}_aff.txt >> ${rigid_list}

   let count=count+1
   moving_name=${name}

   fixed_tensor="--fixed_tensor ${fixed_name}"
   moving_tensor="--moving_tensor ${moving_name}"


    if [ ${start_step} -eq 0 ]   
    then       
      # my_RUN "DRTAMASRigid ${fixed_tensor} ${moving_tensor} --affine_gradient_step_length 1. "
       DRTAMASRigid ${fixed_tensor} ${moving_tensor} --affine_gradient_step_length 1.  &
    fi  
    
    let cnt=cnt+1         
    if [ $cnt -gt 4 ]
    then
        wait
        let cnt=0
    fi
done

wait


if [ ${start_step} -le 0 ] 
    then
       my_RUN "AverageAffineTransformations2  ${rigid_list}   ${subjects}    ${listdir}/average_rigid_0.nii   ${listdir}/average_rigid.txt"
       sed -i 's/MatrixOffsetTransformBase_double_3_3/AffineTransform_double_3_3/g' ${listdir}/average_rigid.txt       
       my_RUN "InvertTransformation ${listdir}/average_rigid.txt"

        curr_id=0
    	for aff in `cat ${rigid_list} `
	    do
            my_RUN "CombineTransformations ${aff} ${listdir}/average_rigid_inv.txt"
            cp ${listdir}/combined_affine.txt $aff

            subj=${subjects_list[$curr_id]}

     	    filename=$(basename "${subj}")
            extension="${filename##*.}"
            filename="${filename%.*}"

            my_RUN "ApplyTransformationToTensor ${subj} $aff   ${listdir}/${filename}_aff.nii ${listdir}/average_rigid_0.nii"
            let curr_id=curr_id+1
        done
fi


if [ ${start_step} -eq 0 ]   
then
    echo "AverageTensors ${subjects_affine} ${listdir}/average_template_rigid.nii 0"
    AverageTensors ${subjects_affine} ${listdir}/average_template_rigid.nii 0
fi







#######################################################INITIAL AFFINE ALIGNMENT ###############################


affine_list=${listdir}/affines.txt

if [ ${start_step} -le 1 ]   
then
    cp ${listdir}/average_template_rigid.nii ${listdir}/average_template_affine_0.nii
fi





let Niter_affine=3


count=0
while [ $count -le ${Niter_affine} ]
do
    if [ -e ${affine_list} ]    
    then 
        rm ${affine_list}
    fi

    fixed_tensor="--fixed_tensor ${listdir}/average_template_affine_${count}.nii"
    

    cnt=0
    
    for (( ns=0; ns<$nsubjects; ns++ ))
    do

        curr_id=$((ns))
        subj=${subjects_list[$curr_id]}
    
        filename=$(basename "${subj}")
        extension="${filename##*.}"
        filename="${filename%.*}"

       	echo ${listdir}/${filename}_aff.txt >> ${affine_list}
      

	    if [ -e "${subj}" ]
        then
            moving_name=${subj}
        else
            moving_name=${listdir}/${subj}
        fi

        moving_tensor="--moving_tensor ${moving_name}"

        if [ ${start_step} -le 1 ] 
        then
            DRTAMAS_cuda ${fixed_tensor} ${moving_tensor}  --only_affine 1 &
        fi
        
        let cnt=cnt+1         
        if [ $cnt -gt 4 ]
        then
            wait
            let cnt=0
        fi
        
    done
    
    wait

    if [ ${start_step} -le 1 ] 
    then      
       my_RUN "AverageAffineTransformations2  ${affine_list}    ${subjects}    ${listdir}/average_template_affine_${count}.nii   ${listdir}/average_affine.txt"
       sed -i 's/MatrixOffsetTransformBase_double_3_3/AffineTransform_double_3_3/g' ${listdir}/average_affine.txt      
       my_RUN "InvertTransformation ${listdir}/average_affine.txt "


        average_affine_name="${listdir}/combined_affine.txt"

        curr_id=0
    	for aff in `cat ${affine_list} `
	    do
            my_RUN "CombineTransformations ${aff} ${listdir}/average_affine_inv.txt"
            cp ${listdir}/combined_affine.txt $aff

            subj=${subjects_list[$curr_id]}

     	    filename=$(basename "${subj}")
            extension="${filename##*.}"
            filename="${filename%.*}"

            my_RUN "ApplyTransformationToTensor ${subj} $aff   ${listdir}/${filename}_aff.nii ${listdir}/average_template_affine_${count}.nii"
            let curr_id=curr_id+1
        done
     fi


    let count=count+1

    if [ ${start_step} -le 1 ] 
    then
        echo "AverageTensors ${subjects_affine} ${listdir}/average_template_affine_${count}.nii 0"
        AverageTensors ${subjects_affine} ${listdir}/average_template_affine_${count}.nii 0
    fi   	
done


if [ -e "${listdir}/combined_affine.txt" ]
then
        rm -rf ${listdir}/combined_affine.txt
fi


#######################################################DIFFEOMORPHIC ALIGNMENT ###############################




aa=0
for aff in `cat ${affine_list}`
do  
    if [ -e ${aff} ]
    then
        name=${aff}
    else
        name=${listdir}/${aff}
    fi

    all_affine_list[$aa]=${name}
    aa=$((aa+1))
done


for ((i=0; i<Nstructurals; i++))
do

   subjects_structurals_aff=`echo $subjects_structurals | sed -e "s/.txt/_aff_${i}.txt/"`

   if [ -e "${subjects_structurals_aff}" ]
   then
        rm -rf ${subjects_structurals_aff}
   fi
done



for subj in `cat ${subjects}`
do
    filename=$(basename "${subj}")
    extension="${filename##*.}"
    filename="${filename%.*}"

	echo ${filename}_diffeo.nii >> ${subjects_diffeo}
    echo ${listdir}/${filename}_def_MINV.nii >> ${deformation_fields}

    for ((i=0; i<Nstructurals; i++))
    do
        subjects_structurals_diffeo=`echo $subjects_structurals | sed -e "s/.txt/_diffeo_${i}.txt/"`
        echo ${filename}_diffeo_${i}.nii >> ${subjects_structurals_diffeo}
    done
done



let last_affine=${Niter_affine}+1

if [ ${diffeomorphic_registration_start_iteration} -eq 0 ]   
then

    if [ ! -z "${subjects_structurals}" ]
    then
        sid=0
        for subj in `cat ${subjects_structurals}`
        do

            all_subjects_structurals[$sid]=${subj}

            IFS=','
            read -ra line_structurals <<< "$subj"
            IFS=$' \t\n'
            for ((i=0; i<Nstructurals; i++))
            do
               curr_structural=${line_structurals[$i]}
               filename=$(basename "${curr_structural}")
               extension="${filename##*.}"
               filename="${filename%.*}"

               cmd="ApplyTransformationToScalar2 ${listdir}/${curr_structural} ${all_affine_list[$sid]} ${listdir}/${filename}_aff.nii ${listdir}/average_template_affine_0.nii BSP"
               my_RUN "${cmd}"

               subjects_structurals_aff=`echo $subjects_structurals | sed -e "s/.txt/_aff_${i}.txt/"`
               echo  ${listdir}/${filename}_aff.nii >> ${subjects_structurals_aff}
            done

            sid=$((sid+1))
        done

        for ((i=0; i<Nstructurals; i++))
        do
            subjects_structurals_aff=`echo $subjects_structurals | sed -e "s/.txt/_aff_${i}.txt/"`
            echo AverageScalars ${subjects_structurals_aff} ${listdir}/average_structural_aff_${i}.nii
            AverageScalars ${subjects_structurals_aff} ${listdir}/average_structural_aff_${i}.nii
        done
    fi

    echo "AverageTensorsWithWeights ${subjects_affine}  ${listdir}/average_template_diffeo_0.nii 0 ${Niter}"
    AverageTensorsWithWeights ${subjects_affine}  ${listdir}/average_template_diffeo_0.nii 0 ${Niter}
    echo "GaussianSmoothTensorImage ${listdir}/average_template_diffeo_0.nii 0.5"
    GaussianSmoothTensorImage ${listdir}/average_template_diffeo_0.nii 0.5
    mv ${listdir}/average_template_diffeo_0_SMTH.nii ${listdir}/average_template_diffeo_0.nii

    for ((i=0; i<Nstructurals; i++))
    do
        cp ${listdir}/average_structural_aff_${i}.nii ${listdir}/average_structural_diffeo_${i}_0.nii
    done
fi



count=${diffeomorphic_registration_start_iteration}
while [ $count -lt ${Niter} ]
do
	echo "Population average creation... Iteration $count"
    template_name=${listdir}/average_template_diffeo_${count}.nii
    
    
    all_structural_templates=""
    if [ ! -z "${subjects_structurals}" ]
    then
        all_structural_templates=${listdir}/"average_structural_diffeo_0_${count}.nii"
        for ((i=1; i<Nstructurals; i++))
        do
            all_structural_templates=${all_structural_templates},${listdir}/average_structural_diffeo_${i}_${count}.nii    
        done
    fi

    

    bash DRTAMAS_batch_register_to_target_main.bash  -t=${template_name} -s=${subjects} -st=${all_structural_templates} -ss=${subjects_structurals} 

    

    if [ ${constrain_deformations} -eq 1 ]   
    then


         echo "AverageDeformationFields ${deformation_fields} ${listdir}/average_def.nii"
         AverageDeformationFields ${deformation_fields} ${listdir}/average_def.nii
         my_RUN "InvertTransformation ${listdir}/average_def.nii"
         av_inv_name=${listdir}/average_def_inv.nii

         sid=0
         for subj in `cat ${subjects}`
         do
             filename=$(basename "${subj}")
             extension="${filename##*.}"
             filename="${filename%.*}"

             my_RUN "CombineTransformations ${listdir}/${filename}_def_MINV.nii ${listdir}/average_def_inv.nii"
             mv ${listdir}/combined_displacement.nii ${listdir}/${filename}_def_MINV_comb.nii
             my_RUN "CombineTransformations ${listdir}/${filename}_aff.txt ${listdir}/${filename}_def_MINV_comb.nii"
             my_RUN "ApplyTransformationToTensor ${listdir}/${subj} ${listdir}/combined_displacement.nii  ${listdir}/${filename}_diffeo.nii ${listdir}/average_template_diffeo_${count}.nii"


             if [ ! -z "${subjects_structurals}" ]
             then
                 line=${all_subjects_structurals[$sid]}
                 IFS=','
                 read -ra line_structurals <<< "$line"
                 IFS=$' \t\n'
                 for ((i=0; i<Nstructurals; i++))
                 do
                      curr_structural=${line_structurals[$i]}
                      my_RUN "ApplyTransformationToScalar2 ${listdir}/${curr_structural} ${listdir}/combined_displacement.nii ${listdir}/${filename}_diffeo_${i}.nii ${listdir}/average_structural_diffeo_${i}_${count}.nii BSP"
                 done
             fi             
             sid=$((sid+1))
         done
    fi



    let count=count+1


    echo "AverageTensorsWithWeights ${subjects_diffeo} ${listdir}/average_template_diffeo_${count}.nii  ${count} ${Niter}"
    AverageTensorsWithWeights ${subjects_diffeo} ${listdir}/average_template_diffeo_${count}.nii  ${count} ${Niter}

    if [ ! -z "${subjects_structurals}" ]
    then
        echo "Averaging scalars...."
        for ((i=0; i<Nstructurals; i++))
        do
            subjects_structurals_diffeo=`echo $subjects_structurals | sed -e "s/.txt/_diffeo_${i}.txt/"`
             echo "AverageScalars ${subjects_structurals_diffeo} ${listdir}/average_structural_diffeo_${i}_${count}"
             AverageScalars ${subjects_structurals_diffeo} ${listdir}/average_structural_diffeo_${i}_${count}
        done
    fi
 
done

echo "DRTAMAS is done....."

