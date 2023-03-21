#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "helper.h"

int main(int argc, char *argv[])
{
    int rank, size;
    int numberOfPictures, numberOfObjects;
    double matchingThreshold;
    int pictureIndex = 0;
    int logsIndex = 0;
    Picture *pictures;
    Object *objects;
    Logs *searchLogs;
    MPI_Status status;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if number of processes is greater than 2
    if (size < 2)
    {
        printf("Number of processes must be greater than 1 for this program to run properly \n");
        MPI_Finalize();
        return 0;
    }
    double startTime = MPI_Wtime();
    // Read input files and allocate memory for logs
    if (rank == 0)
    {
        // read input file
        readInputFile(INPUT_FILE, &pictures, &objects, &matchingThreshold, &numberOfPictures, &numberOfObjects);
        // allocate memory for logs array and initialize it
        searchLogs = (Logs *)malloc(numberOfPictures * sizeof(Logs));
        checkMalloc(searchLogs, "search logs array");

        for (int i = 0; i < numberOfPictures; i++)
        {
            searchLogs[i].pictureID = pictures[i].ID;
            searchLogs[i].numObjectsFound = 0;
            searchLogs[i].objectIDs = NULL;
            searchLogs[i].objectPositions = (Position *)malloc(sizeof(Position) * numberOfObjects);
        }
    }

    // // Broadcast matching threshold, number of pictures, number of objects ans the objects to all processes
    MPI_Bcast(&matchingThreshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numberOfPictures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numberOfObjects, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // send all objects to all processes
        for (int i = 1; i < size; i++)
            for (int j = 0; j < numberOfObjects; j++)
                sendObject(&objects[j], i, OBJECT_TAG);
    }
    else
    {
        objects = (Object *)malloc(numberOfObjects * sizeof(Object));
        checkMalloc(objects, "objects array");

        for (int i = 0; i < numberOfObjects; i++)
            receiveObject(&objects[i], 0, OBJECT_TAG, &status);
    }

    // master process
    if (rank == 0)
    {

        // send each process the first picture to work on
        for (int i = 1; i < size && pictureIndex < numberOfPictures; i++)
        {
            MPI_Send(&pictureIndex, 1, MPI_INT, i, PICTURE_TAG, MPI_COMM_WORLD);
            sendPicture(&pictures[pictureIndex], i, PICTURE_TAG);
            pictureIndex++;
        }

        // while there are pictures to be processed
        while (pictureIndex < numberOfPictures)
        {
            // receive logs from process
            receiveLog(&searchLogs[logsIndex], MPI_ANY_SOURCE, LOGS_TAG, &status);
            logsIndex++;

            // send update picture index to process
            MPI_Send(&pictureIndex, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);

            // send next picture to process
            sendPicture(&pictures[pictureIndex], status.MPI_SOURCE, PICTURE_TAG);
            pictureIndex++;
        }
        // receive logs from all processes
        while (logsIndex < numberOfPictures)
        {
            receiveLog(&searchLogs[logsIndex], MPI_ANY_SOURCE, LOGS_TAG, &status);
            logsIndex++;
        }

        // send terminate signal to all processes
        for (int i = 1; i < size; i++)
            MPI_Send(&pictureIndex, 1, MPI_INT, i, TERMINATE_TAG, MPI_COMM_WORLD);

        // write logs to output file
        writeLogs(OUTPUT_FILE, &searchLogs, numberOfPictures);

        freeLogs(searchLogs, numberOfPictures);
        freePictures(pictures, numberOfPictures);
    }
    else
    {
        MPI_Recv(&pictureIndex, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // while master process does not send terminate signal
        while (status.MPI_TAG != TERMINATE_TAG)
        {
            pictures = (Picture *)malloc(sizeof(Picture));
            // receive first pictures from master process
            receivePicture(pictures, 0, MPI_ANY_TAG, &status);

            // allocate memory for the log
            searchLogs = (Logs *)malloc(sizeof(Logs));
            checkMalloc(searchLogs, "search logs array");
            searchLogs->pictureID = pictures->ID;
            searchLogs->numObjectsFound = 0;
            searchLogs->objectIDs = (int *)malloc(numberOfObjects * sizeof(int));
            checkMalloc(searchLogs->objectIDs, "object IDs array");
            searchLogs->objectPositions = (Position *)malloc(numberOfObjects * sizeof(Position));
            checkMalloc(searchLogs->objectPositions, "object positions array");

            // initialize log positions to -1
            #pragma omp parallel for
            for (int i = 0; i < numberOfObjects; i++)
            {
                searchLogs->objectPositions[i].row = NOT_FOUND;
                searchLogs->objectPositions[i].column = NOT_FOUND;
            }

            // search for objects
            findObjectsInPicture(pictures, objects, searchLogs, numberOfObjects, matchingThreshold);

            // send logs to master process
            sendLog(searchLogs, 0, LOGS_TAG);

            MPI_Recv(&pictureIndex, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            freeLogs(searchLogs, 1);
            freePictures(pictures, 1);
        }
    }

    freeObjects(objects, numberOfObjects);

    double endTime = MPI_Wtime();
    if (rank == 0)
        printf("Time taken: %f \n", endTime - startTime);

    MPI_Finalize();
    return 0;
}
