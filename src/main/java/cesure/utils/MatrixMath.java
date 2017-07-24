package cesure.utils;


public class MatrixMath {

    public static Matrix identity(int size) {
        double[][] newarray = new double[size][size];
        for (int i=0; i<size; i++) {
            for (int j=0; j<size; j++) {
                newarray[i][j] = (i==j) ? 1 : 0;
            }
        }
        return new Matrix(newarray);
    }

    public static Matrix Matrix_transpose(Matrix matrix) {
        double[][] newarray = new double[matrix.nbColumns][matrix.nbRows];
        for (int i = 0; i<matrix.nbRows; i++) {
            for (int j = 0; j<matrix.nbColumns; j++) {
                newarray[j][i] = matrix.array[i][j];
            }
        }
        return new Matrix(newarray);
    }

    public static Matrix Matrix_add(Matrix matrix1, Matrix matrix2) {
        if (matrix1.nbColumns != matrix2.nbColumns || matrix1.nbRows != matrix2.nbRows) {
            throw new MatrixError("Matrix_add(Matrix,Matrix) - "
                    +matrix1.nbRows+" "+matrix1.nbColumns+" "
                    +matrix2.nbRows+" "+matrix2.nbColumns);
        }

        Matrix result = new Matrix(matrix1.nbRows, matrix1.nbColumns);

        for (int i = 0; i<matrix1.nbRows; i++) {
            for (int j = 0; j<matrix1.nbColumns; j++) {
                result.array[i][j] = matrix1.array[i][j] + matrix2.array[i][j];
            }
        }
        return result;
    }

    public static Matrix Matrix_add(Matrix matrix1, double value) {
        Matrix result = new Matrix(matrix1.nbRows, matrix1.nbColumns);

        for (int i = 0; i<matrix1.nbRows; i++) {
            for (int j = 0; j<matrix1.nbColumns; j++) {
                result.array[i][j] = matrix1.array[i][j] + value;
            }
        }
        return result;
    }

    public static Matrix Matrix_substract(Matrix matrix1, Matrix matrix2) {
        if (matrix1.nbColumns != matrix2.nbColumns || matrix1.nbRows != matrix2.nbRows) {
            throw new MatrixError("Matrix_substract(Matrix,Matrix) - "
                    +matrix1.nbRows+" "+matrix1.nbColumns+" "
                    +matrix2.nbRows+" "+matrix2.nbColumns);
        }

        Matrix result = new Matrix(matrix1.nbRows, matrix1.nbColumns);
        for (int i = 0; i<matrix1.nbRows; i++) {
            for (int j = 0; j<matrix1.nbColumns; j++) {
                result.array[i][j] = matrix1.array[i][j] - matrix2.array[i][j];
            }
        }
        return result;
    }

    public static Matrix Matrix_mult(Matrix matrix1, double value) {
        Matrix result = new Matrix(matrix1.nbRows, matrix1.nbColumns);

        for (int i = 0; i<matrix1.nbRows; i++) {
            for (int j = 0; j<matrix1.nbColumns; j++) {
                result.array[i][j] = matrix1.array[i][j] * value;
            }
        }
        return result;
    }

    public static Matrix Matrix_pMult(Matrix matrix1, Matrix matrix2) {
        if (matrix1.nbColumns != matrix2.nbColumns || matrix1.nbRows != matrix2.nbRows) {
            throw new MatrixError("Matrix_pMult(Matrix,Matrix) - m1["+matrix1.nbRows+","+matrix1.nbColumns+"] "
                    +"m2["+matrix2.nbRows+","+matrix2.nbColumns+"]");
        }

        Matrix result = new Matrix(matrix1.nbRows, matrix1.nbColumns);

        for (int i = 0; i<matrix1.nbRows; i++) {
            for (int j = 0; j<matrix1.nbColumns; j++) {
                result.array[i][j] = matrix1.array[i][j] * matrix2.array[i][j];
            }
        }
        return result;
    }

    public static Matrix Matrix_divide(Matrix matrix1, Matrix matrix2) {
        if (matrix1.nbColumns != matrix2.nbColumns || matrix1.nbRows != matrix2.nbRows) {
            throw new MatrixError("Matrix_divide(Matrix,Matrix)");
        }

        Matrix result = new Matrix(matrix1.nbRows, matrix1.nbColumns);

        for (int i = 0; i<matrix1.nbRows; i++) {
            for (int j = 0; j<matrix1.nbColumns; j++) {
                result.array[i][j] = matrix1.array[i][j] / matrix2.array[i][j];
            }
        }
        return result;
    }

    public static Matrix Matrix_mDot(Matrix matrix1, Matrix matrix2) {
        if (matrix1.nbColumns != matrix2.nbRows) {
            throw new MatrixError("Matrix_mDot(Matrix,Matrix) - m1["+matrix1.nbRows+","+matrix1.nbColumns+"] "
                                                              +"m2["+matrix2.nbRows+","+matrix2.nbColumns+"]");
        }


        Matrix result = new Matrix(matrix1.nbRows, matrix2.nbColumns);

        double[][] array1 = matrix1.array;
        double[][] array2 = matrix2.array;
        double[][] resultArray = result.array;

        for (int i = 0; i<matrix1.nbRows; i++) {
            for (int j = 0; j<matrix2.nbColumns; j++) {
                resultArray[i][j] = 0;
                for (int k = 0; k<matrix1.nbColumns; k++) {
                    resultArray[i][j] += array1[i][k] * array2[k][j];
                }
            }
        }
        return result;
    }

    public static double Matrix_vDot(Matrix matrix1, Matrix matrix2) {
        if (matrix1.nbColumns > matrix1.nbRows) {
            matrix1 = Matrix_transpose(matrix1);
        }
        if (matrix2.nbColumns > matrix2.nbRows) {
            matrix2 = Matrix_transpose(matrix2);
        }

        if (matrix1.nbColumns != 1 || matrix2.nbColumns != 1 || matrix1.nbRows != matrix2.nbRows) {
            throw new MatrixError("Matrix_vDot(Matrix,Matrix)");
        }

        double result = 0;
        for (int i = 0; i<matrix1.nbRows; i++) {
            result += matrix1.array[i][0] * matrix2.array[i][0];
        }
        return result;
    }

    public static Matrix Matrix_abs(Matrix matrix) {
        Matrix result = new Matrix(matrix.nbRows, matrix.nbColumns);

        for (int i = 0; i<matrix.nbRows; i++) {
            for (int j = 0; j<matrix.nbColumns; j++) {
                result.array[i][j] = Math.abs(matrix.array[i][j]);
            }
        }
        return result;
    }

    public static double Matrix_mean(Matrix matrix) {
        double result = 0;

        for (int i = 0; i<matrix.nbRows; i++) {
            for (int j = 0; j<matrix.nbColumns; j++) {
                result += matrix.array[i][j];
            }
        }
        result = result / (matrix.nbRows *matrix.nbColumns);
        return result;

    }

    public static Matrix Matrix_deleteCol(final Matrix matrix, final int deleted) {
        if (deleted >= matrix.nbColumns) {
            throw new MatrixError("Can't delete column " + deleted + " from matrix, it only has " + matrix.nbColumns + " columns.");
        }
        final double newMatrix[][] = new double[matrix.nbRows][matrix.nbColumns - 1];

        for (int row = 0; row < matrix.nbRows; row++) {
            int targetCol = 0;
            for (int col = 0; col < matrix.nbColumns; col++) {
                if (col != deleted) {
                    newMatrix[row][targetCol] = matrix.array[row][col]; // get(row, col);
                    targetCol++;
                }
            }

        }
        return new Matrix(newMatrix);
    }

    public static Matrix Matrix_deleteRow(final Matrix matrix, final int deleted) {
        if (deleted >= matrix.nbRows) {
            throw new MatrixError("Can't delete row " + deleted + " from matrix, it only has " + matrix.nbRows + " rows.");
        }
        final double newMatrix[][] = new double[matrix.nbRows - 1][matrix.nbColumns];
        int targetRow = 0;
        for (int row = 0; row < matrix.nbRows; row++) {
            if (row != deleted) {
                for (int col = 0; col < matrix.nbColumns; col++) {
                    newMatrix[targetRow][col] = matrix.array[row][col]; //matrix.get(row, col);
                }
                targetRow++;
            }
        }
        return new Matrix(newMatrix);
    }

    public static Matrix Matrix_concatenateRowMatrix(final Matrix matrixLeft, final Matrix matrixRight) {
        if (!matrixLeft.isRowMatrix() || !matrixRight.isRowMatrix()) {
            throw new MatrixError("Matrix_concatenateRowMatrix(Matrix,Matrix)");
        }

        double[] newArray = new double[matrixLeft.nbColumns+matrixRight.nbColumns];
        int loc = 0;
        for (int i=0; i<matrixLeft.nbColumns; i++) {
            newArray[loc++] = matrixLeft.array[0][i];
        }
        for (int i=0; i<matrixRight.nbColumns; i++) {
            newArray[loc++] = matrixRight.array[0][i];
        }

        return Matrix.newRowMatrix(newArray);
    }
}