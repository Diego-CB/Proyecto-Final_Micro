/**
* --------------------------------------------------------
* Universidad del Valle de Guatemala
* CC3056 - Programación de Microprocesadores
* --------------------------------------------------------
* Proyecto3.cu
* --------------------------------------------------------
* Implementacion de operaciones de reduccion en cuda
* y Calculo de la media en un conjunto de datos.
* --------------------------------------------------------
* Autores:
*   - Diego Cordova,    20212
*   - Alejandro Gomez,  20347
*   - Paola de Leon,    20361
* 
* Fecha de modificacion: 2021/11/23
* --------------------------------------------------------
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 300

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void suma_total(int *a, int *b, int *c, int *d)
{
    int myID = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(myID < *d) 
        c[myID] = a[myID] + b[myID];
}

int main(void) {

    int SIZE_1 = 25000;
    int SIZE_2 = 15000;

    float init_1 = (float) SIZE_1;
    float init_2 = (float) SIZE_2;


    //---------------- Inicializacion de Streams ----------------
    cudaStream_t stream1, stream2;

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);	

    //---------------- Inicializacion de memoria ----------------

    // memoria en host para los streams
    int *a1, *b1, *c1, *d1; 
    int *a2, *b2, *c2, *d2; 

    // memoria en device para los streams
    int *dev_a1, *dev_b1, *dev_c1, *dev_d1;
    int *dev_a2, *dev_b2, *dev_c2, *dev_d2;

    // --------------- Inicializacion de datos ---------------

    cudaHostAlloc((void**)&c1, SIZE_1 * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&c2, SIZE_2 * sizeof(int), cudaHostAllocDefault);
    
    printf("\nGenerando %d numeros aleatorios para el arreglo 1...\n", SIZE_1);
    printf("Generando %d numeros aleatorios para el arreglo 2...\n", SIZE_2);

    int i = 0;
    for(i = 0; i < SIZE_1; i++)
    {
        // Se generan numeros aleatorios entre 100 y 1100
        c1[i] = rand() % 1100 + 100;
    }

    for(i = 0; i < SIZE_2; i++)
    {
        // Se generan numeros aleatorios entre 0 y 3500
        c2[i] = rand() % 3500;
    }

    printf("\nCalculando suma de los dos arreglos...\n");

    while ((SIZE_1 != 1) || (SIZE_2 != 1))
    {
        
        if (SIZE_1 != 1)
        {
            //---------------- Calculo de tamano de nuevo arreglo ----------------

            int limit = 0;

            if (SIZE_1 % 2 == 0)
            {
                limit = SIZE_1 / 2;
            }
            else 
            {
                limit = (SIZE_1 + 1) / 2;
            }

            //---------------- Memory Allocation ----------------

            // Host
            cudaHostAlloc((void**)&a1,limit*sizeof(int), cudaHostAllocDefault);
            cudaHostAlloc((void**)&b1,limit*sizeof(int), cudaHostAllocDefault);
            cudaHostAlloc((void**)&d1,sizeof(int), cudaHostAllocDefault);

            // Device
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
                    if (i < SIZE_1)
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

            SIZE_1 = limit; // Se asigna nuevo tamano de arreglo
            *d1 = limit;    // Se copia el tamano de arreglo a d1

            //Calculo correcto del número de hilos por bloque
            int bloques = SIZE_1 / BLOCKSIZE;

            if(SIZE_1 % BLOCKSIZE != 0)
            {
                bloques = bloques + 1;
            }

            int hilos = BLOCKSIZE;

            // --------------------- Kernel ---------------------

            // Copida de parametros a device
            cudaMemcpyAsync(dev_a1, a1, SIZE_1*sizeof(int), cudaMemcpyHostToDevice, stream1);
            cudaMemcpyAsync(dev_b1, b1, SIZE_1*sizeof(int), cudaMemcpyHostToDevice, stream1);
            cudaMemcpyAsync(dev_d1, d1, sizeof(int), cudaMemcpyHostToDevice, stream1);

            // Ejecucion de Kernel
            suma_total <<< bloques, hilos, 0, stream1 >>> (dev_a1, dev_b1, dev_c1, dev_d1);

            // Copia de resultado a host 
            cudaMemcpyAsync(c1, dev_c1, SIZE_1*sizeof(int), cudaMemcpyDeviceToHost, stream1);
            
            cudaStreamSynchronize(stream1); // wait for stream1 to finish

            // --------------- Liberacion de memoria ---------------

            cudaFree(a1);
            cudaFree(b1);

            cudaFree(dev_a1);
            cudaFree(dev_b1);
            cudaFree(dev_c1);
            cudaFree(dev_d1);
        }

        if (SIZE_2 != 1)
        {
            //---------------- Calculo de tamano de nuevo arreglo ----------------

            int limit = 0;

            if (SIZE_2 % 2 == 0)
            {
                limit = SIZE_2 / 2;
            }
            else 
            {
                limit = (SIZE_2 + 1) / 2;
            }

            //---------------- Memory Allocation ----------------

            // Host
            cudaHostAlloc((void**)&a2, limit * sizeof(int), cudaHostAllocDefault);
            cudaHostAlloc((void**)&b2, limit * sizeof(int), cudaHostAllocDefault);
            cudaHostAlloc((void**)&d2, sizeof(int), cudaHostAllocDefault);

            // Device
            cudaMalloc((void**)&dev_a2, limit * sizeof(int));
            cudaMalloc((void**)&dev_b2, limit * sizeof(int));
            cudaMalloc((void**)&dev_c2, limit * sizeof(int));
            cudaMalloc((void**)&dev_d2, sizeof(int));

            //---------------- Reordenamiento de datos ----------------

            for(i = 0; i < (2 * limit); i++)
            {
                if (i < limit)
                {
                    a2[i] = c2[i];
                }
                else
                {
                    if (i < SIZE_2)
                    {
                        b2[i - limit] = c2[i];
                    }
                    else
                    {
                        b2[i - limit] = 0;
                    }
                }
            }

            // Liberacion parcial de c2 y reasignacion a memoria
            cudaFree(c2);        
            cudaHostAlloc((void**)&c2, limit * sizeof(int), cudaHostAllocDefault);

            SIZE_2 = limit; // Se asigna nuevo tamano de arreglo
            *d2 = limit;    // Se copia el tamano de arreglo a d2

            //Calculo correcto del número de hilos por bloque
            int bloques = SIZE_2 / BLOCKSIZE;

            if(SIZE_2 % BLOCKSIZE != 0)
            {
                bloques = bloques + 1;
            }

            int hilos = BLOCKSIZE;

            // --------------------- Kernel ---------------------

            // Copida de parametros a device
            cudaMemcpyAsync(dev_a2, a2, SIZE_2 * sizeof(int), cudaMemcpyHostToDevice, stream2);
            cudaMemcpyAsync(dev_b2, b2, SIZE_2 * sizeof(int), cudaMemcpyHostToDevice, stream2);
            cudaMemcpyAsync(dev_d2, d2, sizeof(int), cudaMemcpyHostToDevice, stream2);

            // Ejecucion de Kernel
            suma_total <<< bloques, hilos, 0, stream2 >>> (dev_a2, dev_b2, dev_c2, dev_d2);

            // Copia de resultado a host 
            cudaMemcpyAsync(c2, dev_c2, SIZE_2 * sizeof(int), cudaMemcpyDeviceToHost, stream2);

            cudaStreamSynchronize(stream2); // wait for stream2 to finish

            // --------------- Liberacion de memoria ---------------

            cudaFree(a2);
            cudaFree(b2);

            cudaFree(dev_a2);
            cudaFree(dev_b2);
            cudaFree(dev_c2);
            cudaFree(dev_d2);
        }
    }

    // --------------- Impresion de resultados ---------------

    float suma = (float) c1[0];
    float media = suma / init_1;

    printf("\n--------- Arreglo 1 ---------");
    printf("\n-> La suma total de datos es: %d", c1[0]);
    printf("\n-> La media del arreglo 1 es: %lf\n\n", media);

    suma = (float) c2[0];
    media = suma / init_2;

    printf("\n--------- Arreglo 2 ---------");
    printf("\n-> La suma total de datos es: %d", c2[0]);
    printf("\n-> La media del arreglo 2 es: %lf\n\n", media);

    // --------------- Liberacion final de memoria ---------------

    cudaFree(c1);
    cudaFree(d1);
    
    cudaFree(c2);
    cudaFree(d2);

    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream1);

    return 0;
}