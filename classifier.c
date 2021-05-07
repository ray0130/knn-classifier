#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <math.h>
#include "knn.h"

/*****************************************************************************/
/* Do not add anything outside the main function here. Any core logic other  */
/*   than what is described below should go in `knn.c`. You've been warned!  */
/*****************************************************************************/

/**
 * main() takes in the following command line arguments.
 *   -K <num>:  K value for kNN (default is 1)
 *   -d <distance metric>: a string for the distance function to use
 *          euclidean or cosine (or initial substring such as "eucl", or "cos")
 *   -p <num_procs>: The number of processes to use to test images
 *   -v : If this argument is provided, then print additional debugging information
 *        (You are welcome to add print statements that only print with the verbose
 *         option.  We will not be running tests with -v )
 *   training_data: A binary file containing training image / label data
 *   testing_data: A binary file containing testing image / label data
 *   (Note that the first three "option" arguments (-K <num>, -d <distance metric>,
 *   and -p <num_procs>) may appear in any order, but the two dataset files must
 *   be the last two arguments.
 *
 * You need to do the following:
 *   - Parse the command line arguments, call `load_dataset()` appropriately.
 *   - Create the pipes to communicate to and from children
 *   - Fork and create children, close ends of pipes as needed
 *   - All child processes should call `child_handler()`, and exit after.
 *   - Parent distributes the test dataset among children by writing:
 *        (1) start_idx: The index of the image the child should start at
 *        (2)    N:      Number of images to process (starting at start_idx)
 *     Each child should get at most N = ceil(test_set_size / num_procs) images
 *      (The last child might get fewer if the numbers don't divide perfectly)
 *   - Parent waits for children to exit, reads results through pipes and keeps
 *      the total sum.
 *   - Print out (only) one integer to stdout representing the number of test
 *      images that were correctly classified by all children.
 *   - Free all the data allocated and exit.
 *   - Handle all relevant errors, exiting as appropriate and printing error message to stderr
 */
void usage(char *name) {
    fprintf(stderr, "Usage: %s -v -K <num> -d <distance metric> -p <num_procs> training_list testing_list\n", name);
}

int main(int argc, char *argv[]) {

    int opt;
    int K = 1;             // default value for K
    char *dist_metric = "euclidean"; // default distant metric
    int num_procs = 1;     // default number of children to create
    int verbose = 0;       // if verbose is 1, print extra debugging statements
    int total_correct = 0; // Number of correct predictions

    while((opt = getopt(argc, argv, "vK:d:p:")) != -1) {
        switch(opt) {
        case 'v':
            verbose = 1;
            break;
        case 'K':
            K = atoi(optarg);
            break;
        case 'd':
            dist_metric = optarg;
            break;
        case 'p':
            num_procs = atoi(optarg);
            break;
        default:
            usage(argv[0]);
            exit(1);
        }
    }

    if(optind >= argc) {
        fprintf(stderr, "Expecting training images file and test images file\n");
        exit(1);
    }

    char *training_file = argv[optind];
    optind++;
    char *testing_file = argv[optind];

    // TODO The following lines are included to prevent compiler warnings
    // and should be removed when you use the variables.

    // Set which distance function to use
    /* You can use the following string comparison which will allow
     * prefix substrings to match:
     *
     * If the condition below is true then the string matches
     * if (strncmp(dist_metric, "euclidean", strlen(dist_metric)) == 0){
     *      //found a match
     * }
     */

    // TODO
    double (*dist_func)(Image*, Image*);
    if (strncmp(dist_metric, "euclidean", strlen(dist_metric)) == 0){
        //found a match
        dist_func = distance_euclidean;
    }
    else{
        printf("Using cosine");
        dist_func = distance_cosine;
    }

    // Load data sets
    if(verbose) {
        fprintf(stderr,"- Loading datasets...\n");
    }

    Dataset *training = load_dataset(training_file);
    if ( training == NULL ) {
        fprintf(stderr, "The data set in %s could not be loaded\n", training_file);
        exit(1);
    }

    Dataset *testing = load_dataset(testing_file);
    if ( testing == NULL ) {
        fprintf(stderr, "The data set in %s could not be loaded\n", testing_file);
        exit(1);
    }

    // Create the pipes and child processes who will then call child_handler
    if(verbose) {
        printf("- Creating children ...\n");
    }

    // TODO
    int fd[num_procs][2];
    int fd2[num_procs][2];
    int start_i = 0;
    int N = (testing->num_items/num_procs) + 1;
    int r = (testing->num_items)%num_procs;

    for (int i = 0; i < num_procs; i ++) {
        if ((pipe(fd[i])) == -1){
            perror("pipe error");
            exit(1);
        }
        if ((pipe(fd2[i])) == -1){
            perror("pipe2 error");
            exit(1);
        }
        int result = fork();
        if (result < 0){
            // error
            perror("Fork");
            exit(1);
        }
        else if (result == 0){
            // child process
            if (close (fd[i][1]) == -1){
                perror("close unused file descriptors, child");
                exit(1);
            }

            if (close (fd2[i][0]) == -1){
                perror("close unused file descriptors, child");
                exit(1);
            }
            child_handler(training, testing, K, dist_func, fd[i][0], fd2[i][1]);
            free_dataset(training);
            free_dataset(testing);
            exit(0);
        }

        // Distribute the work to the children by writing their starting index and
        // the number of test images to process to their write pipe

        // TODO
        else{
            // parent process
            if (close(fd[i][0]) == -1){
                perror("Close pipe parent");
                exit(1);
            }
            if (close(fd2[i][1]) == -1){
                perror("Close pipe parent");
                exit(1);
            }
            if (i == r){
                N = N - 1;
            }
            if (write(fd[i][1], &N, sizeof(int)) != sizeof(int)) {
                perror("parent write N");
                exit(1);
            }
            if (write(fd[i][1], &start_i, sizeof(int)) != sizeof(int)) {
                perror("parent write start index");
                exit(1);
            }

            if (close (fd[i][1]) == -1){
                perror("close parent writing end");
                exit(1);
            }
            start_i += N;
        }
    }




    // Wait for children to finish
    if(verbose) {
        printf("- Waiting for children...\n");
    }

    // TODO
    for (int i = 0; i < num_procs; i++){

        int status;
        if (wait(&status) == -1){
            perror("Wait");
            exit(1);
        }

        // When the children have finised, read their results from their pipe

        // TODO
        else{
            int c = 0;
            if (read(fd2[i][0], &c, sizeof(int)) != sizeof(int)){
                perror("Parent Read");
                exit(1);
            }
            close(fd2[i][0]);
            total_correct += c;
        }

    }




    if(verbose) {
        printf("Number of correct predictions: %d\n", total_correct);
    }

    // This is the only print statement that can occur outside the verbose check
    printf("%d\n", total_correct);

    // Clean up any memory, open files, or open pipes

    // TODO
    free_dataset(training);
    free_dataset(testing);


    return 0;
}
