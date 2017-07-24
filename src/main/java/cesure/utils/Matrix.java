package cesure.utils;

import java.io.Serializable;
import java.util.Random;

import static cesure.utils.RandomUtils.random;

public class Matrix implements Serializable {

    public int nbRows;
    public int nbColumns;
    public int length;
    public double[][] array;



    public Matrix(int nbRows, int nbColumns) {
        this.nbRows = nbRows;
        this.nbColumns = nbColumns;
        length = nbRows * nbColumns;
        array = new double[nbRows][nbColumns];
    }

    public Matrix(double[][] values) {
        nbRows = values.length;
        nbColumns = values[0].length;
        length = nbRows * nbColumns;
        array = new double[nbRows][nbColumns];
        for (int i = 0; i< nbRows; i++) {
            if (values[i].length != nbColumns) { throw new MatrixError("Matrix(double[][])"); }
            for (int j = 0; j< nbColumns; j++) {
                this.array[i][j] = values[i][j];
            }
        }
    }

    public static Matrix newRowMatrix(int size) {
        return new Matrix(1, size);
    }

    public static Matrix newRowMatrix(double... values) {
        Matrix matrix = new Matrix(1, values.length);
        for (int i=0; i<values.length; i++) {
            matrix.array[0][i] = values[i];
        }
        return matrix;
    }

    public static Matrix newColumnMatrix(int size) {
        return new Matrix(size, 1);
    }

    public static Matrix newColumnMatrix(double[] values) {
        Matrix matrix = new Matrix(values.length, 1);
        for (int i=0; i<values.length; i++) {
            matrix.array[i][0] = values[i];
        }
        return matrix;
    }

    public static Matrix newRandomMatrix(int nbRows, int nbColumns, double min, double max) {
        Matrix matrix = new Matrix(nbRows,nbColumns);
        for (int row=0; row<nbRows; row++) {
            for (int col=0; col<nbColumns; col++) {
                matrix.array[row][col] = random(min,max);
            }
        }
        return matrix;
    }

    public Matrix cp() {
        return new Matrix(array);
    }

    public boolean isRowMatrix() {
        return nbRows == 1;
    }

    public boolean isColumnMatrix() {
        return nbColumns == 1;
    }

    public boolean isVector() { return isRowMatrix() || isColumnMatrix(); }

    public boolean isZero() {
        for (double[] row : array) {
            for (double val : row) {
                if (val != 0) {
                    return false;
                }
            }
        }
        return true;
    }


    public Matrix getRow(int row) {
        if (row < 0 || row > nbRows-1) { throw new MatrixError("getRow(int"); }
        return newRowMatrix(array[row]);
    }

    public Matrix getColumn(int col) {
        if (col < 0 || col > nbColumns-1) { throw new MatrixError("getColumn(int"); }
        double[] newArray = new double[nbRows];
        for (int i=0; i<nbRows; i++) {
            newArray[i] = array[i][col];
        }
        return newColumnMatrix(newArray);
    }


    public double get(int row, int col) {
        if (row < 0 || row >= nbRows || col < 0 || col >= nbColumns) {throw new MatrixError("get(int,int)");}
        return array[row][col];
    }

    public double get(int i) {
        if (i < 0 || i >= length) {throw new MatrixError("get(int)");}
        return array[i%nbRows][i/nbColumns];
    }

    public void set(int i, double value) {
        if (i < 0 || i >= length) {
            throw new MatrixError("set(int,double)");
        }
        array[i%nbRows][i/nbColumns] = value;
    }

    public void set(int row, int column, double value) {
        if (row>=0 && row < nbRows && column >= 0 && column < nbColumns) {
            array[row][column] = value;
        } else {throw new MatrixError("set(int,int,double)");}
    }

    public void setRandom(double min, double max) {
        for (int i = 0; i< nbRows; i++) {
            for (int j = 0; j< nbColumns; j++) {
                array[i][j] = random(min, max);
            }
        }
    }
    public void randomize(double minChange, double maxChange) {
        for (int i = 0; i< nbRows; i++) {
            for (int j = 0; j< nbColumns; j++) {
                array[i][j] += random(minChange, maxChange);
            }
        }
    }
    public void randomize(Random rand, double minChange, double maxChange) {
        for (int i = 0; i< nbRows; i++) {
            for (int j = 0; j< nbColumns; j++) {
                array[i][j] += random(rand, minChange, maxChange);
            }
        }
    }
    public void setZero() {
        for (int i = 0; i< nbRows; i++) {
            for (int j = 0; j< nbColumns; j++) {
                array[i][j] = 0;
            }
        }
    }



    public void print() {
        for (int row = 0; row< nbRows; row++) {
            for (int column = 0; column< nbColumns; column++) {
                if (array[row][column] == 0) {
                    System.out.print("0");
                } else {
                    System.out.print(array[row][column]);
                }
                if (column < nbColumns -1) {System.out.print(", ");}
            }
            System.out.println();
        }
    }
    public void print(String title) {
        System.out.println(title);
        print();
    }


    public Matrix transpose() {
        double[][] newarray = new double[nbColumns][nbRows];
        for (int i = 0; i<nbRows; i++) {
            for (int j = 0; j<nbColumns; j++) {
                newarray[j][i] = array[i][j];
            }
        }
        array = newarray;
        return this;
    }

    public Matrix pow(double pow) {
        for (int rowI=0; rowI<nbRows; rowI++) {
          for (int colI=0; colI<nbColumns; colI++) {
                array[rowI][colI] = Math.pow(array[rowI][colI], pow);
            }
        }
        return this;
    }

    public Matrix add(int row, int col, double value) {
        if (row < 0 || row >= nbRows || col < 0 || col >= nbColumns) {throw new MatrixError("get(int,int)");}
        array[row][col] += value;
        return this;
    }

    public Matrix add(Matrix matrix) {
        if (nbColumns != matrix.nbColumns || nbRows != matrix.nbRows) { throw new MatrixError("add(Matrix"); }
        for (int i = 0; i< nbRows; i++) {
            for (int j = 0; j< nbColumns; j++) {
                array[i][j] += matrix.array[i][j];
            }
        }
        return this;
    }
    public Matrix substract(Matrix matrix) {
        if (nbColumns != matrix.nbColumns || nbRows != matrix.nbRows) { throw new MatrixError("substract(Matrix"); }
        for (int i = 0; i< nbRows; i++) {
            for (int j = 0; j< nbColumns; j++) {
                array[i][j] -= matrix.array[i][j];
            }
        }
        return this;
    }

    public Matrix mult(double a) {
        for (int i = 0; i< nbRows; i++) {
            for (int j = 0; j< nbColumns; j++) {
                array[i][j] *= a;
            }
        }
        return this;
    }
    public Matrix div(double a) {
        for (int i = 0; i< nbRows; i++) {
            for (int j = 0; j< nbColumns; j++) {
                array[i][j] /= a;
            }
        }
        return this;
    }
    public Matrix pMult(Matrix matrix) {
        if (nbColumns != matrix.nbColumns || nbRows != matrix.nbRows) {
            throw new MatrixError("pMult(Matrix) - m1[" + nbRows + "," + nbColumns + "] "
                    + "m2[" + matrix.nbRows + "," + matrix.nbColumns + "]");
        }
        for (int i = 0; i< nbRows; i++) {
            for (int j = 0; j< nbColumns; j++) {
                array[i][j] *= matrix.array[i][j];
            }
        }
        return this;
    }

    public Matrix mDot(Matrix matrix) {
        if (nbColumns != matrix.nbRows) {
            throw new MatrixError("mDot(Matrix,Matrix) - m1["+nbRows+","+nbColumns+"] "
                    +"m2["+matrix.nbRows+","+matrix.nbColumns+"]");
        }
        double[][] newArray = new double[nbRows][matrix.nbColumns];
        for (int i = 0; i<nbRows; i++) {
            for (int j = 0; j<matrix.nbColumns; j++) {
                for (int k = 0; k<nbColumns; k++) {
                    newArray[i][j] += array[i][k] * matrix.array[k][j];
                }
            }
        }
        array = newArray;
        return this;
    }

    public double vDot(Matrix matrix) {
        if (!isVector() || !matrix.isVector() || length != matrix.length) {throw new MatrixError("vDot(Matrix)");}

        double result = 0;
        for (int i = 0; i<length; i++) {
            result += array[i][0] * matrix.array[i][0];
        }
        return result;
    }



    public Matrix gradient() {
        for (int i = 0; i< nbRows; i++) {
            for (int j = 0; j< nbColumns; j++) {
                array[i][j] *= 1 - array[i][j];
            }
        }
        return this;
    }

    public double avg() {
        double moyenne = 0;
        for (int i=0; i<nbRows; i++) {
            for (int j = 0; j< nbColumns; j++) {
                moyenne += array[i][j];
            }
        }
        moyenne = moyenne / length;
        return moyenne;
    }

    public Matrix deleteLastColumn() {
        if (nbColumns < 1) {throw new MatrixError("deleteLastColumn() - "+nbColumns);}

        final double newMatrix[][] = new double[nbRows][nbColumns - 1];
        for (int row = 0; row < nbRows; row++) {
            for (int col = 0; col < nbColumns-1; col++) {
                newMatrix[row][col] = array[row][col];
            }
        }

        return new Matrix(newMatrix);
    }

}