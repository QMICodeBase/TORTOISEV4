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




if [ $# -lt 2 ]
then
    echo 'Usage: DRTAMAS_external_registration_with_constraint  options'
    echo 'OPTIONS:'
    echo '   -s=full_path_to_textfile_with_tensor_images_to_be_registered  or --subjects=full_path_to_textfile_with_tensor_images  (required)'    
    echo '   -t=full_path_to_tensor_template_image  or --template=full_path_to_tensor_template_image  (required)'
    echo '   -st=string_that_contains_all_structural_templates_separated_by_commas or --structural_templates=string_that_contains_all_structural_templates_separated_by_commas '
    echo '   -ss=full_path_to_textfile_with_structural_images_to_be_registered_separated_by_commas_per_line  or --structural_subjects=full_path_to_textfile_with_structural_images_separated_by_commas_per_line  '

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




template=''
subjects=''
structural_subjects=''
structural_templates=''
init=0



for i in "$@"
do
case $i in
    -s=*|--subjects=*)
        subjects="${i#*=}"
        shift 
    ;;
    -t=*|--template=*)
       template="${i#*=}"
        shift 
    ;;
    -ss=*|--structural_subjects=*)
        structural_subjects="${i#*=}"
        shift 
    ;;
    -st=*|--structural_templates=*)
       structural_templates="${i#*=}"
        shift 
    ;;
    -o=*|--output_folder=*)
        output_folder="${i#*=}"
        shift 
    ;;
    -in=*|--in=*)
        init="${i#*=}"
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

if [ ! -e "${template}" ]
then
     echo Template imagefile: ${template} does not exist. Exiting....
     exit
fi



let Nstructurals=0
if [ ! -z "${structural_templates}" ]
then
    IFS=','
    read -ra structurals <<< "${structural_templates}"
    IFS=$' \t\n'
    let Nstructurals=${#structurals[@]}


    for ((i=0; i<Nstructurals; i++))
    do
        if [ ! -e "${structurals[${i}]}" ]
        then
            echo Structural_template file: ${structurals[${i}]} does not exist. Exiting....
            exit
        fi
    done
fi    


if [ !  -z  $output_folder  ]
then
     if [ ! -e "${output_folder}" ]
     then
          mkdir -p  ${output_folder}
     fi
     cp ${subjects} ${output_folder}/
     subjects=${output_folder}/$(basename "${subjects}")
fi


 

listdir=$(dirname "${subjects}")
listdir=`realpath ${listdir}`



# Setting for registering Human Data
fixed_tensor='--fixed_tensor '${template}

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


if [ ! -z "${structural_templates}" ]
then
    aa=0
    for subj in `cat ${structural_subjects}`
    do
        if [ -e ${subj} ]
        then
            name=${subj}
        else
            name=${listdir}/${subj}
        fi

        structural_subjects_list[$aa]=${name}
        aa=$((aa+1))
    done
fi


fixed_scalars=""
for ((i=0; i<Nstructurals; i++))
do
    fixed_scalars="${fixed_scalars} --fixed_anatomical ${structurals[$i]}"
done


if [ -e ${listdir}/temp_registration ]
then
    rm -r ${listdir}/temp_registration
fi



mkdir -p ${listdir}/temp_registration
for (( ns=0; ns<$nsubjects; ns++ ))
do
    curr_id=$((ns))
    subj=${subjects_list[$curr_id]}
    filename=$(basename "${subj}")
    extension="${filename##*.}"        
    filename="${filename%.*}"
    
   
    mv ${subj} `dirname ${subj}`/${filename}_orig.nii
    
    cp `dirname ${subj}`/${filename}_orig.nii  ${listdir}/temp_registration/${filename}.${extension}                        
    echo ${filename}.${extension} >> ${listdir}/temp_registration/subjs.txt
done
cd ${listdir}/temp_registration




cp ${template} ${listdir}/temp_registration/internal_template.nii 


bash DRTAMAS_create_template_nonsmoothlast.bash -s=${listdir}/temp_registration/subjs.txt -c=1
DRTAMAS_cuda -f ${listdir}/temp_registration/internal_template.nii -m ${listdir}/temp_registration/average_template_diffeo_6.nii --only_rigid 1
mv ${listdir}/temp_registration/average_template_diffeo_6_aff.nii ${listdir}/temp_registration/external_template.nii


echo external_template.nii > templates.txt
echo internal_template.nii >> templates.txt


rm ${listdir}/temp_registration/average_template*

bash DRTAMAS_create_template_nonsmoothlast.bash -s=${listdir}/temp_registration/templates.txt -c=1 -i=${listdir}/temp_registration/internal_template.nii 


for (( ns=0; ns<$nsubjects; ns++ ))
do
    curr_id=$((ns))
    subj=${subjects_list[$curr_id]}
    filename=$(basename "${subj}")
    extension="${filename##*.}"        
    filename="${filename%.*}"
   

    mv ${listdir}/${filename}_orig.nii ${listdir}/${filename}.nii
   
    echo "CombineTransformationsWithOutputName -out ${listdir}/temp_registration/${filename}_aff_def_MINV_cnstr_final.nii  -trans ${listdir}/temp_registration/${filename}_aff_def_MINV_cnstr.nii ${listdir}/temp_registration/external_template_aff_def_MINV_cnstr.nii" 
    CombineTransformationsWithOutputName -out ${listdir}/temp_registration/${filename}_aff_def_MINV_cnstr_final.nii  -trans ${listdir}/temp_registration/${filename}_aff_def_MINV_cnstr.nii ${listdir}/temp_registration/external_template_aff_def_MINV_cnstr.nii
    
    echo ApplyTransformationToTensor ${subj} ${listdir}/temp_registration/${filename}_aff_def_MINV_cnstr_final.nii ${listdir}/temp_registration/${filename}_diffeo_final.nii ${template}
    ApplyTransformationToTensor ${subj} ${listdir}/temp_registration/${filename}_aff_def_MINV_cnstr_final.nii ${listdir}/temp_registration/${filename}_diffeo_final.nii ${template}                
done




echo "Done registering subjects ..."

