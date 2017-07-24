package cesure.utils;

import java.util.Random;

public class RandomUtils {

    public static double random(Random rand, double min, double max) {
        return min + (max-min) * rand.nextDouble();
    }

    public static double random(double min, double max) {
        return min + (max-min) * Math.random();
    }

    public static double random(double range, boolean abs) {
        return abs ? random(0,range) : random(-range, range);
    }

}
