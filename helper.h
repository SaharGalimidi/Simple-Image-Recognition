#pragma once
#include <mpi.h>

#define INPUT_FILE "input.txt"
#define OUTPUT_FILE "output.txt"
#define PICTURE_TAG 0
#define OBJECT_TAG 1
#define LOGS_TAG 2
#define TERMINATE_TAG 3
#define THREADS_PER_BLOCK 1024
#define NOT_FOUND -1

struct PictureStruct
{
    int ID;
    int dimension;
    int *colorsMatrix;
};
typedef struct PictureStruct Picture;

struct ObjectStruct
{
    int ID;
    int dimension;
    int *subColorsMatrix;
};
typedef struct ObjectStruct Object;

struct PositionStruct
{
    int row;
    int column;
};
typedef struct PositionStruct Position;

struct LogsStruct
{
    int pictureID;
    int numObjectsFound;
    int *objectIDs;
    Position *objectPositions;
};
typedef struct LogsStruct Logs;

// -----------------------Service Functions---------------------------

/*
 * This function frees the pictures array
 * @param pictures: the pictures array
 * @param numPictures: the number of pictures
 * @return: void
 */
void freePictures(Picture *pictures, int numPictures);

/*
 * This function frees the objects array
 * @param objects: the objects array
 * @param numObjects: the number of objects
 * @return: void
 */
void freeObjects(Object *objects, int numObjects);

/*
 * This function frees the logs array
 * @param logs: the logs array
 * @param numLogs: the number of logs
 * @return: void
 */
void freeLogs(Logs *logs, int numLogs);

/*
 * This function checks if the fscanf function succeeded
 * @param read: the number of read items
 * @param expected: the expected number of read items
 * @param message: the message to print if the read failed
 * @return: void
 */
void checkRead(int read, int expected, const char *message);

/*
 * This function checks if the malloc function succeeded
 * @param ptr: the pointer to the allocated memory
 * @param message: the message to print if the malloc failed
 * @return: void
 */
void checkMalloc(void *ptr, const char *message);

/*
 * This function reads the colors matrix from the input file
 * @param fp: the input file pointer
 * @param colorsMatrix: the colors matrix
 * @param dimension: the dimension of the matrix
 * @return: void
 */
void readColorsMatrix(FILE *fp, int *colorsMatrix, int dimension);

/*
 * This function reads the pictures from the input file
 * @param fp: the input file pointer
 * @param pictures: pointer to the array of pictures
 * @param numberOfPictures: the number of pictures
 * @return: void
 */
void readPictures(FILE *fp, Picture **pictures, int *numberOfPictures);

/*
 * This function reads the objects from the input file
 * @param fp: the input file pointer
 * @param objects: pointer to the array of objects
 * @param numberOfObjects: the number of objects
 * @return: void
 */
void readObjects(FILE *fp, Object **objects, int *numberOfObjects);

/*
 * This function is used to read all the input data from the input file
 * @param inputFile: the input file name
 * @param pictures: the array of pictures
 * @param objects: the array of objects
 * @param matchingThreshold: the matching threshold
 * @param numberOfPictures: the number of pictures
 * @param numberOfObjects: the number of objects
 * @return: void
 */
void readInputFile(const char *inputFile, Picture **pictures, Object **objects, double *matchingThreshold, int *numberOfPictures, int *numberOfObjects);

/*
 * This function writes the logs to the output file
 * @param outputFile: the output file name
 * @param logs: the array of logs
 * @param numberOfLogs: the number of logs
 * @return: void
 */
void writeLogs(const char *outputFile, Logs **logs, int numberOfLogs);

// ---------------------- MPI Functions -------------------------------

/*
 * This function sends a picture to a specific rank
 * @param picture: the picture
 * @param destRank: the destination rank
 * @param tag: the tag
 * @return: void
 */
void sendPicture(Picture *picture, int destRank, int tag);

/*
 * This function receives a picture from a specific rank
 * @param picture: the picture
 * @param sourceRank: the source rank
 * @param tag: the tag
 * @param status: the status
 * @return: void
 */
void receivePicture(Picture *picture, int sourceRank, int tag, MPI_Status *status);

/*
 * This function sends an object to a specific rank
 * @param object: the object
 * @param destRank: the destination rank
 * @param tag: the tag
 * @return: void
 */
void sendObject(Object *object, int destRank, int tag);

/*
 * This function receives an object from a specific rank
 * @param object: the object
 * @param sourceRank: the source rank
 * @param tag: the tag
 * @param status: the status
 * @return: void
 */
void receiveObject(Object *object, int sourceRank, int tag, MPI_Status *status);

/*
 * This function sends Logs to a specific rank
 * @param logs: the logs
 * @param destRank: the destination rank
 * @param tag: the tag
 * @return: void
 */
void sendLog(Logs *log, int destRank, int tag);

/*
 * This function receives Logs from a specific rank
 * @param logs: the logs
 * @param sourceRank: the source rank
 * @param tag: the tag
 * @param status: the status
 * @return: void
 */
void receiveLog(Logs *log, int sourceRank, int tag, MPI_Status *status);

// ---------------------- OpenMP Functions -------------------------------
/*
 * This function calculates the matching between a picture and an object
 * @param picture: pointer to the picture
 * @param object: array of objects to be found in the picture
 * @param log: the log of the picture and the objects found in it
 * @param matching: the matching threshold
 * @return: void
 */
void findObjectsInPicture(Picture *picture, Object *objects, Logs *log, int numberOfObjects, double matchingThreshold);

// ---------------------- CUDA Functions ---------------------------------

/*
 * This function calculates the matching between a picture and an object
 * @param picture: pointer to the picture
 * @param object: array of objects to be found in the picture
 * @param matching: the matching between the picture and the object
 * @param upperLeftCornerRow: the upper left corner row of the object in the picture
 * @param upperLeftCornerCol: the upper left corner column of the object in the picture
 * @return: void
 */
extern void calculateMatchingOnGPU(Picture *picture, Object *object, int *upperLeftCorner, double matchingThreshold);
