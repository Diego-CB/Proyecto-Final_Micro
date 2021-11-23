/**
* --------------------------------------------------------
* Universidad del Valle de Guatemala
* CC3056 - Programación de Microprocesadores
* --------------------------------------------------------
* Proyecto2.cu
* --------------------------------------------------------
* Implementacion de operaciones de reduccion en cuda
* y Calculo de la media y porcentaje en un conjunto 
* de datos.
* --------------------------------------------------------
* Autores:
*   - Diego Cordova, 20212
*   - Alejandro Gomez, 20347
*   - Paola de Leon, 20361
* 
* Fecha de modificacion: 2021/11/22
* --------------------------------------------------------
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 300

// Arreglo con valores de i
// Arreglo con resultado de iteracion

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void suma_total(int *a, int *b, int *c, int *d)
{
    int myID = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(myID < *d) 
        c[myID] = a[myID] + b[myID];
}

int main(void) {

    int SIZE1 = 15000;
    //int SIZE2 = 15000;

    //--- Stream management ---
    //Object creation
    cudaStream_t stream1, stream2;
    //Stream initialization
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);	

    //---------------- Inicializacion de memoria ----------------

    // memoria en host para los streams
    int *a1, *b1, *c1, *d1; 

    // memoria en device para los streams
    int *dev_a1, *dev_b1, *dev_c1, *dev_d1;

    // --------------- Inicializacion de datos ---------------

    cudaHostAlloc((void**)&c1,SIZE1*sizeof(int), cudaHostAllocDefault);

    int i = 0;
    int s = 0;
    int temp;

    // Stream 1
    for(i = 0; i < SIZE1; i++)
    {
        temp = rand() % 1100 + 100;

        c1[i] = temp;
        s += temp;
    }

    printf("\nSuma esperada: %d\n\n", s);


    while (SIZE1 != 1)
    {

        //---------------- Calculo de tamano de nuevo arreglo ----------------

        int limit = 0;

        if (SIZE1 % 2 == 0)
        {
            limit = SIZE1 / 2;
        }
        else 
        {
            limit = (SIZE1 + 1) / 2;
        }

        //---------------- Memory Allocation ----------------

        //stream 1 - pinned memory allocation
        //---- pinned memory is exclusively physical. No paging to disk
        //---- Faster but limited.  Use with caution
        cudaHostAlloc((void**)&a1,limit*sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc((void**)&b1,limit*sizeof(int), cudaHostAllocDefault);
        cudaHostAlloc((void**)&d1,sizeof(int), cudaHostAllocDefault);

        cudaMalloc((void**)&dev_a1, limit * sizeof(int));
        cudaMalloc((void**)&dev_b1, limit * sizeof(int));
        cudaMalloc((void**)&dev_c1, limit * sizeof(int));
        cudaMalloc((void**)&dev_d1, sizeof(int));

        //---------------- Reordenamiento de datos ----------------

        for(i = 0; i < (2 * limit); i++)
        {
            if (i < limit)
            {
                a1[i] = c1[i];
            }
            else
            {
                if (i < SIZE1)
                {
                    b1[i - limit] = c1[i];
                }
                else
                {
                    b1[i - limit] = 0;
                }
            }
        }

        // Liberacion parcial de c1 y reasignacion a memoria
        cudaFree(c1);        
        cudaHostAlloc((void**)&c1,limit*sizeof(int), cudaHostAllocDefault);

        SIZE1 = limit; // Se asigna nuevo tamano de arreglo
        *d1 = limit;    // Se copia el tamano de arreglo a d1

        //Calculo correcto del número de hilos por bloque
        int bloques = SIZE1 / BLOCKSIZE;

        if(SIZE1 % BLOCKSIZE != 0)
        {
            bloques = bloques + 1;
        }

        int hilos = BLOCKSIZE;

        // --------------------- Kernel ---------------------

        // Copida de parametros a device
        cudaMemcpyAsync(dev_a1, a1, SIZE1*sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_b1, b1, SIZE1*sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_d1, d1, sizeof(int), cudaMemcpyHostToDevice, stream1);

        // Ejecucion de Kernel
        suma_total <<< bloques, hilos, 0, stream1 >>> (dev_a1, dev_b1, dev_c1, dev_d1);

        // Copia de resultado a host 
        cudaMemcpyAsync(c1, dev_c1, SIZE1*sizeof(int), cudaMemcpyDeviceToHost, stream1);
        
        cudaStreamSynchronize(stream1); // wait for stream1 to finish
        printf("->: %d\n", c1[0]);

        // --------------- Liberacion de memoria ---------------

        cudaFree(a1);
        cudaFree(b1);

        cudaFree(dev_a1);
        cudaFree(dev_b1);
        cudaFree(dev_c1);
        cudaFree(dev_d1);
    }

    // --------------- Impresion de resultados ---------------

    int suma = c1[0];
    printf("\nSuma total: %d\n", suma);

    // --------------- Liberacion final de memoria ---------------

    cudaFree(c1);
    cudaFree(d1);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}