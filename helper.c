#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "helper.h"

void freePictures(Picture *pictures, int numPictures)
{
    for (int i = 0; i < numPictures; i++)
        free(pictures[i].colorsMatrix);
    free(pictures);
}

void freeObjects(Object *objects, int numObjects)
{
    for (int i = 0; i < numObjects; i++)
        free(objects[i].subColorsMatrix);
    free(objects);
}

void freeLogs(Logs *logs, int numLogs)
{
    for (int i = 0; i < numLogs; i++)
    {
        free(logs[i].objectIDs);
        free(logs[i].objectPositions);
    }
    free(logs);
}

void checkRead(int read, int expected, const char *message)
{
    if (read != expected)
    {
        printf("Error reading %s \r \n", message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void checkMalloc(void *ptr, const char *message)
{
    if (ptr == NULL)
    {
        printf("Error allocating memory for %s \r \n", message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void readColorsMatrix(FILE *fp, int *colorsMatrix, int dimension)
{
    for (int j = 0; j < dimension; j++)
        for (int k = 0; k < dimension; k++)
            checkRead(fscanf(fp, "%d", &colorsMatrix[j * dimension + k]), 1, "color");
}

void readPictures(FILE *fp, Picture **pictures, int *numberOfPictures)
{
    // read number of pictures
    checkRead(fscanf(fp, "%d", numberOfPictures), 1, "number of pictures");

    // allocate memory for pictures array
    *pictures = (Picture *)malloc(*numberOfPictures * sizeof(Picture));
    checkMalloc(*pictures, "pictures array");

    // read pictures from file and store them in pictures array of Picture structs
    for (int i = 0; i < *numberOfPictures; i++)
    {
        // read picture ID
        checkRead(fscanf(fp, "%d", &(*pictures)[i].ID), 1, "picture ID");

        // read picture dimension
        checkRead(fscanf(fp, "%d", &(*pictures)[i].dimension), 1, "picture dimension");

        // allocate memory for colors matrix
        (*pictures)[i].colorsMatrix = (int *)malloc((*pictures)[i].dimension * (*pictures)[i].dimension * sizeof(int));
        checkMalloc((*pictures)[i].colorsMatrix, "colors matrix of picture");
        readColorsMatrix(fp, (*pictures)[i].colorsMatrix, (*pictures)[i].dimension);
    }
}

void readObjects(FILE *fp, Object **objects, int *numberOfObjects)
{
    // read number of objects
    checkRead(fscanf(fp, "%d", numberOfObjects), 1, "number of objects");

    // allocate memory for objects array
    *objects = (Object *)malloc(*numberOfObjects * sizeof(Object));
    checkMalloc(*objects, "objects array");

    // read objects from file and store them in objects array of Object structs
    for (int i = 0; i < *numberOfObjects; i++)
    {
        // read object ID
        checkRead(fscanf(fp, "%d", &(*objects)[i].ID), 1, "object ID");

        // read object dimension
        checkRead(fscanf(fp, "%d", &(*objects)[i].dimension), 1, "object dimension");

        // allocate memory for colors matrix
        (*objects)[i].subColorsMatrix = (int *)malloc((*objects)[i].dimension * (*objects)[i].dimension * sizeof(int));
        checkMalloc((*objects)[i].subColorsMatrix, "colors matrix of object");
        readColorsMatrix(fp, (*objects)[i].subColorsMatrix, (*objects)[i].dimension);
    }
}

void readInputFile(const char *inputFile, Picture **pictures, Object **objects, double *matchingThreshold, int *numberOfPictures, int *numberOfObjects)
{
    FILE *fp = fopen(inputFile, "r");
    checkMalloc(fp, "file pointer");
    checkRead(fscanf(fp, "%lf", matchingThreshold), 1, "matching threshold");
    readPictures(fp, pictures, numberOfPictures);
    readObjects(fp, objects, numberOfObjects);
}

void writeLogs(const char *outputFile, Logs **logs, int numberOfLogs)
{
    // Open file
    FILE *fp = fopen(outputFile, "w");
    checkMalloc(fp, "file pointer");

    // write logs to file
    for (int i = 0; i < numberOfLogs; i++)
    {
        if ((*logs)[i].numObjectsFound < 3)
            fprintf(fp, "Picture %d: No three different Objects were found\r\n", (*logs)[i].pictureID);
        else
        {
            fprintf(fp, "Picture %d: found Objects: ", (*logs)[i].pictureID);
            for (int j = 0; j < (*logs)[i].numObjectsFound; j++)
                if ((*logs)[i].objectPositions[j].row != -1 && (*logs)[i].objectPositions[j].column != -1)
                    fprintf(fp, " %d Position(%d,%d);", (*logs)[i].objectIDs[j], (*logs)[i].objectPositions[j].row, (*logs)[i].objectPositions[j].column);
            fprintf(fp, "\r\n");
        }
    }
    fclose(fp);
}

void sendPicture(Picture *picture, int destRank, int tag)
{
    MPI_Send(&picture->ID, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD);
    MPI_Send(&picture->dimension, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD);
    MPI_Send(picture->colorsMatrix, picture->dimension * picture->dimension, MPI_INT, destRank, tag, MPI_COMM_WORLD);
}

void receivePicture(Picture *picture, int sourceRank, int tag, MPI_Status *status)
{
    MPI_Recv(&picture->ID, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);
    MPI_Recv(&picture->dimension, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);
    picture->colorsMatrix = (int *)malloc(picture->dimension * picture->dimension * sizeof(int));
    checkMalloc(picture->colorsMatrix, "colors matrix of picture");
    MPI_Recv(picture->colorsMatrix, picture->dimension * picture->dimension, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);
}

void sendLog(Logs *log, int destRank, int tag)
{
    MPI_Send(&log->pictureID, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD);
    MPI_Send(&log->numObjectsFound, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD);
    MPI_Send(log->objectIDs, log->numObjectsFound, MPI_INT, destRank, tag, MPI_COMM_WORLD);
    for (int i = 0; i < log->numObjectsFound; i++)
    {
        MPI_Send(&log->objectPositions[i].row, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD);
        MPI_Send(&log->objectPositions[i].column, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD);
    }
}

void receiveLog(Logs *log, int sourceRank, int tag, MPI_Status *status)
{
    MPI_Recv(&log->pictureID, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);
    MPI_Recv(&log->numObjectsFound, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);
    log->objectIDs = (int *)malloc(log->numObjectsFound * sizeof(int));
    checkMalloc(log->objectIDs, "object IDs of picture");
    log->objectPositions = (Position *)malloc(log->numObjectsFound * sizeof(Position));
    checkMalloc(log->objectPositions, "object positions of picture");
    MPI_Recv(log->objectIDs, log->numObjectsFound, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);
    for (int i = 0; i < log->numObjectsFound; i++)
    {
        MPI_Recv(&log->objectPositions[i].row, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);
        MPI_Recv(&log->objectPositions[i].column, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);
    }
}

void sendObject(Object *object, int destRank, int tag)
{
    MPI_Send(&object->ID, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD);
    MPI_Send(&object->dimension, 1, MPI_INT, destRank, tag, MPI_COMM_WORLD);
    MPI_Send(object->subColorsMatrix, object->dimension * object->dimension, MPI_INT, destRank, tag, MPI_COMM_WORLD);
}

void receiveObject(Object *object, int sourceRank, int tag, MPI_Status *status)
{
    MPI_Recv(&object->ID, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);
    MPI_Recv(&object->dimension, 1, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);
    object->subColorsMatrix = (int *)malloc(object->dimension * object->dimension * sizeof(int));
    checkMalloc(object->subColorsMatrix, "colors matrix of object");
    MPI_Recv(object->subColorsMatrix, object->dimension * object->dimension, MPI_INT, sourceRank, tag, MPI_COMM_WORLD, status);
}

void findObjectsInPicture(Picture *picture, Object *objects, Logs *log, int numberOfObjects, double matchingThreshold)
{
    #pragma omp parallel num_threads(numberOfObjects)
    {
        #pragma omp single
        {
            for (int i = 0; i < numberOfObjects; i++)
            {
                #pragma omp task firstprivate(i)
                {
                    int upperLeftCorner = NOT_FOUND;
                    // calculate the matching value for each possible position of the object in the picture using CUDA
                    calculateMatchingOnGPU(picture, objects + i, &upperLeftCorner, matchingThreshold);
                    if (upperLeftCorner != NOT_FOUND)
                    {
                        #pragma omp critical
                        {
                            log->pictureID = picture->ID;
                            log->objectIDs[log->numObjectsFound] = objects[i].ID;
                            log->objectPositions[log->numObjectsFound].row = upperLeftCorner / picture->dimension;
                            log->objectPositions[log->numObjectsFound].column = upperLeftCorner % picture->dimension;
                            log->numObjectsFound++;
                        }
                    }
                }
            }
        }
    }
}
