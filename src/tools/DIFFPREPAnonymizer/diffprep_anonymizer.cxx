
#include "diffprep_anonymizer.h"

#include <sstream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <boost/endian/conversion.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/filesystem.hpp>


#include "itkImportImageFilter.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>


#include <gdcmTrace.h>
#include <itkMetaDataObject.h>


#define kDICOMStr 64

struct TagPairs
{
    std::string tag1;
    std::string tag2;
};



std::vector<TagPairs> getDefaultTags()
{
    std::string tag1,tag2;
    TagPairs tags;
    std::vector<TagPairs> all_tags;

    tag1="0008";
    tag2="0050";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0008";
    tag2="0080";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0008";
    tag2="0081";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0008";
    tag2="0090";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0008";
    tag2="1010";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0008";
    tag2="1030";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0008";
    tag2="103E";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0008";
    tag2="1040";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0008";
    tag2="1048";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0008";
    tag2="1050";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0008";
    tag2="1070";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0010";
    tag2="0010";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);



    tag1="0010";
    tag2="0020";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0010";
    tag2="0021";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0010";
    tag2="0030";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0010";
    tag2="0040";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0010";
    tag2="1010";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0010";
    tag2="1020";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0010";
    tag2="1030";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0010";
    tag2="1040";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0010";
    tag2="2110";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0010";
    tag2="2150";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);



    tag1="0018";
    tag2="1000";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0020";
    tag2="000D";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0020";
    tag2="000E";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0020";
    tag2="0010";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0032";
    tag2="1032";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);




    tag1="0032";
    tag2="1033";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);
    tag1="0032";
    tag2="1060";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);



    tag1="0040";
    tag2="0280";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);


    tag1="0008";
    tag2="0100";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);

    tag1="0008";
    tag2="010F";
    tags.tag1=tag1;tags.tag2=tag2;all_tags.push_back(tags);





    return all_tags;

}



std::vector<TagPairs> ParseTags(char *fname)
{
    TagPairs tags;
    std::vector<TagPairs> all_tags;

    std::ifstream infile(fname);
    std::string line;
    while (std::getline(infile, line))
    {
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace),line.end());
        if(line!=std::string(""))
        {
            int pos= line.find(',');
            std::string tag1= line.substr(0,pos);
            std::string tag2= line.substr(pos+1);

            tags.tag1=tag1;
            tags.tag2=tag2;
            all_tags.push_back(tags);
        }
    }
    infile.close();
    return all_tags;
}



void AddIfNotPresent(std::vector<fs::path> & series_directory, fs::path candidate_path)
{
    bool mexists=false;
    for(int i=0;i<series_directory.size();i++)
    {
        if(series_directory[i].string() == candidate_path.string())
        {
            mexists=true;
            break;
        }
    }

    if(mexists==false)
    {
        series_directory.push_back(candidate_path);
    }
}


bool CheckIfDICOM(std::string fname)
{
    ImageIOType::Pointer dicomIO = ImageIOType::New();
    dicomIO->SetLoadPrivateTags(true);
    gdcm::Trace::SetWarning(false);
    typedef itk::ImageFileReader<SliceType> SliceReaderType;
    SliceReaderType::Pointer reader= SliceReaderType::New();
    reader->SetImageIO(dicomIO);

    return dicomIO->CanReadFile(fname.c_str());
}


std::vector<fs::path> getDICOMPaths(fs::path input_path)
{
    std::vector<fs::path> series_directory;


    fs::recursive_directory_iterator it(input_path);
    fs::recursive_directory_iterator endit;

    while(it != endit)
    {
        bool is_dicom = CheckIfDICOM(it->path().string());
        if(is_dicom)
        {
             AddIfNotPresent(series_directory, it->path().parent_path());
        }
        ++it;
    }
    return series_directory;
}



int isDICOMfile(const char * fname) { //0=NotDICOM, 1=DICOM, 2=Maybe(not Part 10 compliant)
    FILE *fp = fopen(fname, "rb");
 if (!fp)  return 0;
 fseek(fp, 0, SEEK_END);
 long fileLen=ftell(fp);
    if (fileLen < 256) {
        fclose(fp);
        return 0;
    }
 fseek(fp, 0, SEEK_SET);
 unsigned char buffer[256];
 size_t sz = fread(buffer, 1, 256, fp);
 fclose(fp);
 if (sz < 256) return 0;
    if ((buffer[128] == 'D') && (buffer[129] == 'I')  && (buffer[130] == 'C') && (buffer[131] == 'M'))
     return 1; //valid DICOM
    if ((buffer[0] == 8) && (buffer[1] == 0)  && (buffer[3] == 0))
     return 2; //not valid Part 10 file, perhaps DICOM object
    return 0;
} //i


int dcmStrLen (int len) {
    if (len < kDICOMStr)
        return len+1;
    else
        return kDICOMStr;
} //dcmStrLen()


void dcmStr(int lLength, unsigned char lBuffer[], char* lOut) {
    //char test[] = " 1     2    3    ";
    //lLength = (int)strlen(test);

    if (lLength < 1) return;
//#ifdef _MSC_VER
 char * cString = (char *)malloc(sizeof(char) * (lLength + 1));
//#else
//	char cString[lLength + 1];
//#endif
    cString[lLength] =0;
    memcpy(cString, (char*)&lBuffer[0], lLength);
    //memcpy(cString, test, lLength);
    //printMessage("X%dX\n", (unsigned char)d.patientName[1]);
    for (int i = 0; i < lLength; i++)
        //assume specificCharacterSet (0008,0005) is ISO_IR 100 http://en.wikipedia.org/wiki/ISO/IEC_8859-1
        if (cString[i]< 1) {
            unsigned char c = (unsigned char)cString[i];
            if ((c >= 192) && (c <= 198)) cString[i] = 'A';
            if (c == 199) cString[i] = 'C';
            if ((c >= 200) && (c <= 203)) cString[i] = 'E';
            if ((c >= 204) && (c <= 207)) cString[i] = 'I';
            if (c == 208) cString[i] = 'D';
            if (c == 209) cString[i] = 'N';
            if ((c >= 210) && (c <= 214)) cString[i] = 'O';
            if (c == 215) cString[i] = 'x';
            if (c == 216) cString[i] = 'O';
            if ((c >= 217) && (c <= 220)) cString[i] = 'O';
            if (c == 221) cString[i] = 'Y';
            if ((c >= 224) && (c <= 230)) cString[i] = 'a';
            if (c == 231) cString[i] = 'c';
            if ((c >= 232) && (c <= 235)) cString[i] = 'e';
            if ((c >= 236) && (c <= 239)) cString[i] = 'i';
            if (c == 240) cString[i] = 'o';
            if (c == 241) cString[i] = 'n';
            if ((c >= 242) && (c <= 246)) cString[i] = 'o';
            if (c == 248) cString[i] = 'o';
            if ((c >= 249) && (c <= 252)) cString[i] = 'u';
            if (c == 253) cString[i] = 'y';
            if (c == 255) cString[i] = 'y';
        }
    for (int i = 0; i < lLength; i++)
        if ((cString[i]<1) || (cString[i]==' ') || (cString[i]==',') || (cString[i]=='^') || (cString[i]=='/') || (cString[i]=='\\')  || (cString[i]=='%') || (cString[i]=='*')) cString[i] = '_';
    int len = 1;
    for (int i = 1; i < lLength; i++) { //remove repeated "_"
        if ((cString[i-1]!='_') || (cString[i]!='_')) {
            cString[len] =cString[i];
            len++;
        }
    } //for each item
    if (cString[len-1] == '_') len--;
    //while ((len > 0) && (cString[len]=='_')) len--; //remove trailing '_'
    cString[len] = 0; //null-terminate, strlcpy does this anyway
    len = dcmStrLen(len);
    if (len == kDICOMStr) { //we need space for null-termination
  if (cString[len-2] == '_') len = len -2;
 }
    memcpy(lOut,cString,len-1);
    lOut[len-1] = 0;
//#ifdef _MSC_VER
 free(cString);
//#endif
} //dcmStr()


void AnonymizeDICOMFILE(std::string infile, std::string outfile,std::vector<TagPairs> tags )
{


#define  kUnused 0x0001+(0x0001 << 16 )
#define  kStart 0x0002+(0x0000 << 16 )
#define  kTransferSyntax 0x0002+(0x0010 << 16)
//#define  kSpecificCharacterSet 0x0008+(0x0005 << 16 ) //someday we should handle foreign characters...
#define  kImageTypeTag 0x0008+(0x0008 << 16 )
#define  kStudyDate 0x0008+(0x0020 << 16 )
#define  kAcquisitionDate 0x0008+(0x0022 << 16 )
#define  kAcquisitionDateTime 0x0008+(0x002A << 16 )
#define  kStudyTime 0x0008+(0x0030 << 16 )
#define  kAcquisitionTime 0x0008+(0x0032 << 16 )
#define  kManufacturer 0x0008+(0x0070 << 16 )
#define  kInstitutionName 0x0008+(0x0080 << 16 )
#define  kInstitutionAddress 0x0008+(0x0081 << 16 )
#define  kReferringPhysicianName 0x0008+(0x0090 << 16 )
#define  kSeriesDescription 0x0008+(0x103E << 16 ) // '0008' '103E' 'LO' 'SeriesDescription'
#define  kManufacturersModelName 0x0008+(0x1090 << 16 )
#define  kDerivationDescription 0x0008+(0x2111 << 16 )
#define  kComplexImageComponent (uint32_t) 0x0008+(0x9208 << 16 )//'0008' '9208' 'CS' 'ComplexImageComponent'
#define  kPatientName 0x0010+(0x0010 << 16 )
#define  kPatientID 0x0010+(0x0020 << 16 )
#define  kBodyPartExamined 0x0018+(0x0015 << 16)
#define  kScanningSequence 0x0018+(0x0020 << 16)
#define  kSequenceVariant 0x0018+(0x0021 << 16)
#define  kMRAcquisitionType 0x0018+(0x0023 << 16)
#define  kSequenceName 0x0018+(0x0024 << 16)
#define  kZThick  0x0018+(0x0050 << 16 )
#define  kTR  0x0018+(0x0080 << 16 )
#define  kTE  0x0018+(0x0081 << 16 )
#define  kTI  0x0018+(0x0082 << 16) // Inversion time
#define  kEchoNum  0x0018+(0x0086 << 16 ) //IS
#define  kMagneticFieldStrength  0x0018+(0x0087 << 16 ) //DS
#define  kZSpacing  0x0018+(0x0088 << 16 ) //'DS' 'SpacingBetweenSlices'
#define  kPhaseEncodingSteps  0x0018+(0x0089 << 16 ) //'IS'
#define  kEchoTrainLength  0x0018+(0x0091 << 16 ) //IS
#define  kDeviceSerialNumber  0x0018+(0x1000 << 16 ) //LO
#define  kSoftwareVersions  0x0018+(0x1020 << 16 ) //LO
#define  kProtocolName  0x0018+(0x1030<< 16 )
#define  kRadionuclideTotalDose  0x0018+(0x1074<< 16 )
#define  kRadionuclideHalfLife  0x0018+(0x1075<< 16 )
#define  kRadionuclidePositronFraction  0x0018+(0x1076<< 16 )
#define  kGantryTilt  0x0018+(0x1120  << 16 )
#define  kXRayExposure  0x0018+(0x1152  << 16 )
#define  kAcquisitionMatrix  0x0018+(0x1310  << 16 ) //US
#define  kFlipAngle  0x0018+(0x1314  << 16 )
#define  kInPlanePhaseEncodingDirection  0x0018+(0x1312<< 16 ) //CS
#define  kPatientOrient  0x0018+(0x5100<< 16 )    //0018,5100. patient orientation - 'HFS'
//#define  kDiffusionBFactorSiemens  0x0019+(0x100C<< 16 ) //   0019;000C;SIEMENS MR HEADER  ;B_value
#define  kLastScanLoc  0x0019+(0x101B<< 16 )
#define  kDiffusionDirectionGEX  0x0019+(0x10BB<< 16 ) //DS
#define  kDiffusionDirectionGEY  0x0019+(0x10BC<< 16 ) //DS
#define  kDiffusionDirectionGEZ  0x0019+(0x10BD<< 16 ) //DS
#define  kBandwidthPerPixelPhaseEncode  0x0019+(0x1028<< 16 ) //FD
#define  kStudyID 0x0020+(0x0010 << 16 )
#define  kSeriesNum 0x0020+(0x0011 << 16 )
#define  kAcquNum 0x0020+(0x0012 << 16 )
#define  kImageNum 0x0020+(0x0013 << 16 )
#define  kStudyInstanceUID 0x0020+(0x000D << 16 )
#define  kSeriesInstanceUID 0x0020+(0x000E << 16 )
#define  kPatientPosition 0x0020+(0x0032 << 16 )
#define  kOrientationACR 0x0020+(0x0035 << 16 )
#define  kOrientation 0x0020+(0x0037 << 16 )
#define  kImagesInAcquisition 0x0020+(0x1002 << 16 ) //IS
#define  kImageComments 0x0020+(0x4000<< 16 )// '0020' '4000' 'LT' 'ImageComments'
#define  kLocationsInAcquisitionGE 0x0021+(0x104F<< 16 )// 'SS' 'LocationsInAcquisitionGE'
#define  kSamplesPerPixel 0x0028+(0x0002 << 16 )
#define  kPlanarRGB 0x0028+(0x0006 << 16 )
#define  kDim3 0x0028+(0x0008 << 16 ) //number of frames - for Philips this is Dim3*Dim4
#define  kDim2 0x0028+(0x0010 << 16 )
#define  kDim1 0x0028+(0x0011 << 16 )
#define  kXYSpacing  0x0028+(0x0030 << 16 ) //'0028' '0030' 'DS' 'PixelSpacing'
#define  kBitsAllocated 0x0028+(0x0100 << 16 )
#define  kBitsStored 0x0028+(0x0101 << 16 )//'0028' '0101' 'US' 'BitsStored'
#define  kIsSigned 0x0028+(0x0103 << 16 )
#define  kIntercept 0x0028+(0x1052 << 16 )
#define  kSlope 0x0028+(0x1053 << 16 )
#define  kGeiisFlag 0x0029+(0x0010 << 16 ) //warn user if dreaded GEIIS was used to process image
#define  kCSAImageHeaderInfo  0x0029+(0x1010 << 16 )
#define  kCSASeriesHeaderInfo 0x0029+(0x1020 << 16 )
    //#define  kObjectGraphics  0x0029+(0x1210 << 16 )    //0029,1210 syngoPlatformOOGInfo Object Oriented Graphics
#define  kProcedureStepDescription 0x0040+(0x0254 << 16 )
#define  kRealWorldIntercept  0x0040+uint32_t(0x9224 << 16 ) //IS dicm2nii's SlopInt_6_9
#define  kRealWorldSlope  0x0040+uint32_t(0x9225 << 16 ) //IS dicm2nii's SlopInt_6_9
#define  kDiffusionBFactorGE  0x0043+(0x1039 << 16 ) //IS dicm2nii's SlopInt_6_9
#define  kCoilSiemens  0x0051+(0x100F << 16 )
#define  kImaPATModeText  0x0051+(0x1011 << 16 )
#define  kLocationsInAcquisition  0x0054+(0x0081 << 16 )
#define  kDoseCalibrationFactor  0x0054+(0x1322<< 16 )
#define  kIconImageSequence 0x0088+(0x0200 << 16 )
#define  kDiffusionBFactor  0x2001+(0x1003 << 16 )// FL
#define  kSliceNumberMrPhilips 0x2001+(0x100A << 16 ) //IS Slice_Number_MR
#define  kNumberOfSlicesMrPhilips 0x2001+(0x1018 << 16 )//SL 0x2001, 0x1018 ), "Number_of_Slices_MR"
#define  kSliceOrient  0x2001+(0x100B << 16 )//2001,100B Philips slice orientation (TRANSVERSAL, AXIAL, SAGITTAL)
//#define  kLocationsInAcquisitionPhilips  0x2001+(0x1018 << 16 ) //
//#define  kStackSliceNumber  0x2001+(0x1035 << 16 )//? Potential way to determine slice order for Philips?
#define  kNumberOfDynamicScans  0x2001+(0x1081 << 16 )//'2001' '1081' 'IS' 'NumberOfDynamicScans'
#define  kMRAcquisitionTypePhilips 0x2005+(0x106F << 16)
#define  kAngulationAP 0x2005+(0x1071 << 16)//'2005' '1071' 'FL' 'MRStackAngulationAP'
#define  kAngulationFH 0x2005+(0x1072 << 16)//'2005' '1072' 'FL' 'MRStackAngulationFH'
#define  kAngulationRL 0x2005+(0x1073 << 16)//'2005' '1073' 'FL' 'MRStackAngulationRL'
#define  kMRStackOffcentreAP 0x2005+(0x1078 << 16)
#define  kMRStackOffcentreFH 0x2005+(0x1079 << 16)
#define  kMRStackOffcentreRL 0x2005+(0x107A << 16)
#define  kPhilipsSlope 0x2005+(0x100E << 16 )
#define  kDiffusionDirectionRL 0x2005+(0x10B0 << 16)
#define  kDiffusionDirectionAP 0x2005+(0x10B1 << 16)
#define  kDiffusionDirectionFH 0x2005+(0x10B2 << 16)
#define  k2005140F 0x2005+(0x140F << 16)
#define  kWaveformSq 0x5400+(0x0100 << 16)
#define  kImageStart 0x7FE0+(0x0010 << 16 )
#define  kImageStartFloat 0x7FE0+(0x0008 << 16 )
#define  kImageStartDouble 0x7FE0+(0x0009 << 16 )
#define kNest 0xFFFE +(0xE000 << 16 ) //Item follows SQ
#define  kUnnest  0xFFFE +(0xE00D << 16 ) //ItemDelimitationItem [length defined] http://www.dabsoft.ch/dicom/5/7.5/
#define  kUnnest2 0xFFFE +(0xE0DD << 16 )//SequenceDelimitationItem [length undefined]
#define kDICOMStr 64
#define kMANUFACTURER_UNKNOWN  0
#define kMANUFACTURER_SIEMENS  1
#define kMANUFACTURER_GE  2
#define kMANUFACTURER_PHILIPS  3
#define kMANUFACTURER_TOSHIBA  4

    const char *fname = infile.c_str();
    const char *fname2 = outfile.c_str();

    struct stat s;
    if( stat(fname,&s) == 0 )
    {
        if( !(s.st_mode & S_IFREG) )
        {
           printf( "DICOM read fail: not a valid file (perhaps a directory) %s\n",fname);
            return ;
        }
    }

    bool isPart10prefix = true;
    int isOK = isDICOMfile(fname);
    if (isOK == 2)
    {
     isPart10prefix = false;
    }

    FILE *file = fopen(fname, "rb");
    if (!file)
    {
        printf("Unable to open file %s\n", fname);
        return;
    }

    fseek(file, 0, SEEK_END);
    long fileLen=ftell(file); //Get file length

    size_t MaxBufferSz = fileLen;

    if (MaxBufferSz > fileLen)
        MaxBufferSz = fileLen;
    long lFileOffset = 0;
    fseek(file, 0, SEEK_SET);

    unsigned char *buffer=(unsigned char *)malloc(MaxBufferSz+1);

    size_t sz = fread(buffer, 1, MaxBufferSz, file);

    fclose(file);


    uint32_t lLength;
    uint32_t groupElement;
    long lPos = 0;
    bool isSwitchToImplicitVR=false;
    bool isExplicitVR=true;
    int sqDepth = 0;
    int nest=0;

    if (isPart10prefix) { //for part 10 files, skip preamble and prefix
        lPos = 128+4; //4-byte signature starts at 128
        groupElement = buffer[lPos] | (buffer[lPos+1] << 8) | (buffer[lPos+2] << 16) | (buffer[lPos+3] << 24);
    }

    char vr[2];


    while ( ((lPos+8+lFileOffset) <  fileLen))
    {

        groupElement = buffer[lPos] | (buffer[lPos+1] << 8) | (buffer[lPos+2] << 16) | (buffer[lPos+3] << 24);


     /*   unsigned long long mm1, mm2;
        mm1=0x0008;
        mm2=0x1110;
        unsigned long long mm= mm1+ (mm2 << 16);

        if(groupElement==mm)
            int ma=0;
            */

        if ((isSwitchToImplicitVR) && ((groupElement & 0xFFFF) != 2))
        {
            isSwitchToImplicitVR = false;
            isExplicitVR = false;
        }
        lPos += 4;

        if (((groupElement == kNest) || (groupElement == kUnnest) || (groupElement == kUnnest2)) )
        {
            vr[0] = 'N';
            vr[1] = 'A';
            if (groupElement == kUnnest2)
                sqDepth--;

            lLength = 4;
        }
        else
        {
            if (isExplicitVR)
            {
                vr[0] = buffer[lPos];
                vr[1] = buffer[lPos+1];
                if (buffer[lPos+1] < 'A')
                {

                    lLength = buffer[lPos] | (buffer[lPos+1] << 8) | (buffer[lPos+2] << 16) | (buffer[lPos+3] << 24);
                    lPos += 4;
                }
                else
                {
                    if ( ((buffer[lPos] == 'U') && (buffer[lPos+1] == 'N'))
                           || ((buffer[lPos] == 'U') && (buffer[lPos+1] == 'T'))
                           || ((buffer[lPos] == 'O') && (buffer[lPos+1] == 'B'))
                           || ((buffer[lPos] == 'O') && (buffer[lPos+1] == 'W'))
                           )
                    {
                        lPos = lPos + 4;  //skip 2 byte VR string and 2 reserved bytes = 4 bytes

                        lLength = buffer[lPos] | (buffer[lPos+1] << 8) | (buffer[lPos+2] << 16) | (buffer[lPos+3] << 24);
                        lPos = lPos + 4;  //skip 4 byte length
                    }
                    else if   ((buffer[lPos] == 'S') && (buffer[lPos+1] == 'Q'))
                        {
                            lLength = 8; //Sequence Tag
                            //lLength = buffer[lPos] | (buffer[lPos+1] << 8) | (buffer[lPos+2] << 16) | (buffer[lPos+3] << 24);

                        }
                        else
                        { //explicit VR with 16-bit length

                            lLength = buffer[lPos+2] | (buffer[lPos+3] << 8);
                            lPos += 4;  //skip 2 byte VR string and 2 length bytes = 4 bytes
                        }

                }
            }
            else //implicit VR
            {
                vr[0] = 'N';
                vr[1] = 'A';
                lLength = buffer[lPos] | (buffer[lPos+1] << 8) | (buffer[lPos+2] << 16) | (buffer[lPos+3] << 24);
                lPos += 4;  //we have loaded the 32-bit length
            }
        }

        if (lLength == 0xFFFFFFFF)
        {
            lLength = 8; //SQ (Sequences) use 0xFFFFFFFF [4294967295] to denote unknown length
            vr[0] = 'S';
            vr[1] = 'Q';
            long tmp_pos=lPos;
            long cnt=0;
            long tmp_ge=0;
            tmp_ge = buffer[tmp_pos] | (buffer[tmp_pos+1] << 8) | (buffer[tmp_pos+2] << 16) | (buffer[tmp_pos+3] << 24);
            while(tmp_ge!=kUnnest2)
            {
                tmp_pos++;
                cnt++;
                tmp_ge = buffer[tmp_pos] | (buffer[tmp_pos+1] << 8) | (buffer[tmp_pos+2] << 16) | (buffer[tmp_pos+3] << 24);
            }
            lLength=cnt;

        }




        if(groupElement==kTransferSyntax)
        {
            char transferSyntax[kDICOMStr];
            dcmStr (lLength, &buffer[lPos], transferSyntax);

            if (strcmp(transferSyntax, "1.2.840.10008.1.2") == 0)
                isSwitchToImplicitVR = true;

        }



        for(int t=0;t<tags.size();t++)
        {
            std::string tag1= tags[t].tag1;
            std::string tag2= tags[t].tag2;

            unsigned long long tag1n,tag2n;
            std::stringstream ss,ss2;
            ss << std::hex << tag1.c_str();
            ss >> tag1n;
            ss2 << std::hex << tag2.c_str();
            ss2 >> tag2n;

            unsigned long long total_tag= tag1n+ (tag2n << 16);

            if(groupElement==total_tag)
            {
                if(t==0)
                    int mm=0;
                memset(buffer+lPos,'X',lLength);
            }

        }

        lPos = lPos + (lLength);


    } //while


    FILE *file2 = fopen(fname2, "wb");
    if (!file2)
    {
        printf("Unable to open file %s\n", fname2);
        return;
    }

    fwrite(buffer, 1, MaxBufferSz, file2);
    fclose(file2);


}


void AnonymizeFolder(fs::path input_path, fs::path output_path, fs::path dicom_folder,std::vector<TagPairs> tags )
{

    fs::path relative_folder_path = fs::relative(dicom_folder,input_path);

    fs::path new_folder_path = output_path / relative_folder_path;

    if(!fs::exists(new_folder_path))
        fs::create_directories(new_folder_path);


    ImageIOType::Pointer dicomIO = ImageIOType::New();
    dicomIO->SetLoadPrivateTags(true);
    gdcm::Trace::SetWarning(false);
    typedef itk::ImageFileReader<SliceType> SliceReaderType;
    SliceReaderType::Pointer reader= SliceReaderType::New();
    reader->SetImageIO(dicomIO);

    fs::directory_iterator it(dicom_folder);
    fs::directory_iterator endit;

    while(it != endit)
    {
        bool is_dicom = dicomIO->CanReadFile(it->path().string().c_str());
        if(is_dicom)
        {
            fs::path relative_file_path = fs::relative(it->path(),input_path);
            fs::path new_path = output_path / relative_file_path;


            AnonymizeDICOMFILE(it->path().string(),new_path.string(),tags);
        }


        ++it;
    }

}



int main(int argc, char *argv[])
{
    if(argc==1)
    {
        std::cout<<"Usage: DIFFPREPAnonymizer path_to_parent_DICOM_folder path_to_output_parent_DICOM_folder path_to_textfile_containing_tags_to_be_anonymized (optional)."<<std::endl;
        return EXIT_FAILURE;
    }

    std::vector<TagPairs> tags;

    if(argc<4)
    {
        tags= getDefaultTags();
    }
    else
    {
        tags= ParseTags(argv[3]);
    }


    std::string input_folder(argv[1]);
    if(input_folder.at(input_folder.length()-1)==boost::filesystem::path::preferred_separator)
        input_folder= input_folder.substr(0,input_folder.length()-1);

    fs::path input_path(input_folder);
    fs::path output_path(argv[2]);


    if(!fs::exists(input_path))
    {
        std::cout<<"Input DICOM folder does not exist...Exiting..."<<std::endl;
        return EXIT_FAILURE;
    }


    std::vector<fs::path> DICOM_folder_paths= getDICOMPaths(input_path);


    for(int i=0;i<DICOM_folder_paths.size();i++)
    {
        AnonymizeFolder(input_path, output_path, DICOM_folder_paths[i],tags );
    }





}



