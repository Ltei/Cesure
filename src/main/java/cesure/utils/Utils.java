package cesure.utils;

public class Utils {



    public static double[] concatenate(double[] a, double[] b) {
        double[] res = new double[a.length+b.length];
        int loc=0;
        for (double d : a) { res[loc++] = d; }
        for (double d : b) { res[loc++] = d; }
        return res;
    }


}
