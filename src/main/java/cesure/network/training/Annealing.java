package cesure.network.training;

import cesure.network.Cesure;
import cesure.network.CesureMusic;

import java.util.Random;

public class Annealing {

    /****************************************************************
     * A thread class used in the train() method
     ****************************************************************/
    private static class AnnealingThread2 extends Thread {

        private final Random rand;

        private final NetworkAndError source;

        private final CesureMusic music;
        private final int start;
        private final double magnitude;

        private NetworkAndError output;

        /****************************************************************
         * Constructor
         * @param source The source network
         * @param music The music to compute network's error on
         * @param start See SimulatedAnnealing.java
         * @param magnitude The magnitude for each child
         ****************************************************************/
        private AnnealingThread2(Random rand, NetworkAndError source, CesureMusic music, int start, double magnitude) {
            this.rand = rand;
            this.source = source;
            this.music = music;
            this.start = start;
            this.magnitude = magnitude;
        }

        @Override
        public void run() {
            Cesure newNetwork = new Cesure(source.network, rand, magnitude);
            output = new NetworkAndError(newNetwork, newNetwork.calculateErrorSum(music,start));
        }
    }


    /****************************************************************
     * Perform a genetic annealing training on one music
     * Use multiThreading : each processors will compute possibility
     * @param rand The Random object to use
     * @param network The Cesure object to train
     * @param music The CesureMusic object to train on
     * @param start The first note to start guess at :
     *              The "start" first notes will change the context
     *              without being trained on
     * @param magnitude The magnitude
     * @param iterations The number of iterations
     ****************************************************************/
    public static void train(Random rand, Cesure network, CesureMusic music, int start, double magnitude, int iterations) {

        final int nbProcessors = Runtime.getRuntime().availableProcessors();

        NetworkAndError bestNetwork = new NetworkAndError(network, network.calculateErrorSum(music,start));

        Random[] threadRandoms = new Random[nbProcessors];
        for (int i=0; i<nbProcessors; i++) {
            threadRandoms[i] = new Random(rand.nextLong());
        }

        for (int epochI=0; epochI<iterations; epochI++) {
            final double actualMagnitude = (iterations-epochI) * magnitude / iterations;
            final AnnealingThread2[] threads = new AnnealingThread2[nbProcessors];

            for (int threadI=0; threadI<nbProcessors; threadI++) {
                threads[threadI] = new AnnealingThread2(threadRandoms[threadI], bestNetwork, music, start, actualMagnitude);
                threads[threadI].start();
            }
            try {
                for (int threadI=0; threadI<nbProcessors; threadI++) {
                    threads[threadI].join();
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }


            for (int threadI=0; threadI<nbProcessors; threadI++) {
                if (threads[threadI].output.error_sum < bestNetwork.error_sum) {
                    bestNetwork = threads[threadI].output;
                }
            }

            System.out.print("Epoch #"+epochI+" - Error = "+bestNetwork.error_sum);
            for (AnnealingThread2 thread : threads) {
                System.out.print(" ("+thread.output.error_sum+")");
            }
            System.out.println();
        }

        network.setGates(bestNetwork.network);
    }

}
