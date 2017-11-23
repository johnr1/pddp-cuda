#include "../include/file_io.h"

double* file_read(char *filename,int  *numCoords,int  *numObjs) 
{
    double *objects;
    int     i, j, len;
    ssize_t numBytesRead;
    int done=0; 
    FILE *infile;
    char *line, *ret;
    int lineLen;
    
	//don't skip lines or attributes for this project
	int lines_to_skip=0; 
	int attr_to_skip=0;
	
    if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            return NULL;
    }

    /* first find the number of objects */
    lineLen = MAX_CHAR_PER_LINE;
    line = (char*) malloc(lineLen);
    assert(line != NULL);

    (*numCoords) = 0;

     while (fgets(line, lineLen, infile) != NULL) {
            /* check each line to find the max line length */
            while (strlen(line) == lineLen-1) {
                /* this line read is not complete */
                len = strlen(line);
                fseek(infile, -len, SEEK_CUR);

                /* increase lineLen */
                lineLen += MAX_CHAR_PER_LINE;
                line = (char*) realloc(line, lineLen);
                assert(line != NULL);

                ret = fgets(line, lineLen, infile);
                assert(ret != NULL);
            }

            if (strtok(line, " \t\n") != 0)
                (*numCoords)++;
    }
    
    (*numCoords)-=lines_to_skip;
    
     if((*numCoords)<=0)
     {
            fprintf(stderr, "Error: No objects found\n");
            return NULL;
     }
    
	rewind(infile);
      
	/*find the number of attributes*/  
    (*numObjs)=0;

    fgets(line, lineLen, infile);
    
    char * pch;
    pch=strtok(line, ",;");
    
	while (pch != NULL )
	{

		pch = strtok (NULL, ",;");
		(*numObjs)++;
	}
    
    if(attr_to_skip!=NONE)
    {
      (*numObjs)--;
      if(attr_to_skip==BOTH)
          (*numObjs)--;
    }
        
    rewind(infile);


    /* allocate space for objects and read all objects */
    len = (*numCoords) * (*numObjs);
    objects    = (double*)malloc( len * sizeof(double));
    assert(objects != NULL);



    /* read all objects */
           
    for(i=0;i<lines_to_skip;i++)
       fgets(line, lineLen, infile);
    
    i=0;
	j=0;

    while (fgets(line, lineLen, infile) != NULL) 
	{
             pch=strtok(line, ",;");
             while (pch != NULL && j<(*numObjs))
			 {
                if(attr_to_skip%2==1 && j==0 && done==0)
                {
                      done=1;
                      pch = strtok (NULL, ",;");
                      continue;                      
                }
                objects[i*(*numObjs)+j]=atof(pch);
                pch = strtok (NULL, ",;");
				j++;
			 }
			 i++;
			 j=0;
			 done=0;
    }
    
	assert(i == *numCoords);

    fclose(infile);
    free(line);
    

    return objects;
}

void print(Matrix A) {
    printf("Matrix [%dx%d] = \n", A.rows, A.cols);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            printf("\t%e", A.matrix[i * A.cols + j]);
        }
        printf("\n");
    }
}

void print_to_file(Matrix A, char *filename) {
    FILE *fp = fopen(filename, "w");
    for (int i = 0; i < A.rows; i++) {
        fprintf(fp, "%f", A.matrix[i * A.cols]);
        for (int j = 1; j < A.cols; j++) {
            fprintf(fp, "\t%.15f", A.matrix[i * A.cols + j]);
        }
        fprintf(fp, "\n");
    }
}