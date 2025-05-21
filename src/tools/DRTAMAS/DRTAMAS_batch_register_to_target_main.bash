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
    echo 'Usage: DRTAMAS_batch_register_to_target_with  options'
    echo 'OPTIONS:'
    echo '   -t=full_path_to_tensor_template_image  or --template=full_path_to_tensor_template_image  (required)'
    echo '   -s=full_path_to_textfile_with_tensor_images  or --subjects=full_path_to_textfile_with_tensor_images  (required)'
    echo '   -st=string_that_contains_all_structural_templates_separated_by_commas or --structural_templates=string_that_contains_all_structural_templates_separated_by_commas '
    echo '   -ss=full_path_to_textfile_with_structural_images_separated_by_commas_per_line  or --structural_subjects=full_path_to_textfile_with_structural_images_separated_by_commas_per_line  '
    echo '   -o=output_folder   or --output_folder=full_path_to_output_folder (optional. if not provided, files are placed in the folder of the textfile of images. )'

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
output_folder=''



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


gcnt=0
gpuid=0

for (( ns=0; ns<$nsubjects; ns++ ))
do
    curr_id=$((ns))
    subj=${subjects_list[$curr_id]}


    filename=$(basename "${subj}")
    extension="${filename##*.}"
    filename="${filename%.*}"
   

    if [ -e "${subj}" ]
    then
        moving_name=${subj}
    else
        moving_name=${listdir}/${subj}
    fi

    moving_scalars=""
    if [ ! -z "${structural_templates}" ]
    then
        IFS=','
        read -ra line_structurals <<< "${structural_subjects_list[$curr_id]}"
        IFS=$' \t\n'

        for ((i=0; i<Nstructurals; i++))
        do
           curr_structural=${line_structurals[$i]}
           if [ ! -e "${curr_structural}" ]
           then
               curr_structural=${listdir}/${curr_structural}
           fi
           
           moving_scalars="${moving_scalars} --moving_anatomical ${curr_structural}"
        done
    fi

    moving_tensor="--moving_tensor ${moving_name}"
    
    #DRTAMAS_cuda ${fixed_tensor}  ${moving_tensor} ${fixed_scalars} ${moving_scalars} --step 1 
              
    my_RUN "DRTAMAS_cuda ${fixed_tensor}  ${moving_tensor} ${fixed_scalars} ${moving_scalars} --step 1"
done

