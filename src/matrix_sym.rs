use std::ops::{DerefMut, Deref};
use std::fmt;

use crate::matrix_asym as masy;

/* Symmetric matrix of generic size */
#[derive(Copy, Clone)]
pub struct MatrixSym<const M: usize>(pub masy::MatrixAsym<M,M>);


impl<const M: usize> MatrixSym<M>{
    pub fn new() -> Self{
        Self { 0: masy::MatrixAsym::new() }
    }

    /* Perform the matrix operation b*M*(b^t) 
       This operation gets a seperate function because mathematically it alway returns a symmetric matrix.
       If b.mult(M.mult(&transpose(&b))) was used, the result could be asymmetric due to rounding errors.*/
    pub fn b_mult_self_mult_bt<const U: usize>(&self, b: &masy::MatrixAsym<M,U>) -> MatrixSym<U>{
        let mut result = MatrixSym::new();
        let mut tmp = [[0.0; M]; U];
        let mut sum;

        /* Left multiplication */
        for i in 0..U {
            for j in 0..M {
                sum = 0.0;
                for k in 0..M {
                    sum += b[i][k] * self[k][j];
                }
                tmp[i][j] = sum;
            }
        }

        /* Right multiplication and symmetrization */
        for i in 0..U {
            for j in 0..=i {
                sum = 0.0;
                for k in 0..M {
                    sum += tmp[i][k] * b[j][k];
                }
                result[i][j] = sum;
                result[j][i] = sum;
            }
        }
        result
    }

    /* Perform the Cholesky decomposition */
    pub fn chol(&self) -> Result<masy::MatrixAsym<M,M>, &'static str>{
        let mut sum;
        let mut result = masy::MatrixAsym::new();

        for i in 0..M{
            for j in 0..=i{

                sum = 0.0;
                for k in 0..j{
                    sum += result[i][k] * result[j][k];
                }

                if i == j{
                    if self[i][i] - sum >= 0.0 {
                        result[i][j] = (self[i][i] - sum).sqrt();
                    }else {
                        return Err("Negative squareroot");
                    }
                } else {
                    if result[j][j] != 0.0{
                        result[i][j] = (1.0 / result[j][j]) * (self[i][j] - sum);
                    }else{
                        return Err("Div 0");
                    }
                }
            }
        }
        Ok(result)
    }

    /* Solve the equation A*x = b using Cholesky decomposition
       A has to be symmetric and positive definite */
    pub fn chol_solve<const U: usize>(&self, b: &masy::MatrixAsym<U,M>) -> Result<masy::MatrixAsym<U,M>, &'static str>{
        let mut result = masy::MatrixAsym::new();
        let mut m_y = [[0.0;U];M];
        let mut sum;

        /* Perform Cholesky decomposition */
        let m_l = self.chol()?;

        /* forward substitution */
        for i in 0..U{
            for j in 0..M{
                sum = 0.0;
                for k in 0..j{
                    sum += m_y[k][i] * m_l[j][k];
                }
                if m_l[j][j] != 0.0{
                    m_y[j][i] = (b[j][i]-sum)/m_l[j][j];
                }else{
                    return Err("Div 0");
                }
            }
        }

        /* back substitution */
        for i in 0..U{
            for j in (0..M).rev(){
                sum = 0.0;
                for k in (j..M).rev(){
                    sum += result[k][i] * m_l[k][j];
                }
                if m_l[j][j] != 0.0{
                    result[j][i] = (m_y[j][i]-sum)/m_l[j][j];
                }else{
                    return Err("Div 0");
                }
            }
        }
        Ok(result)
    }

    /* subtract two matrices of the same size */
    pub fn sub(&mut self, b: &MatrixSym<M>) -> &mut Self{
        for i in 0..M{
            for j in 0..=i{
                self[i][j] -= b[i][j];
                self[j][i] = self[i][j];
            }
        }
        self
    }

    /* add two matrices of the same size */
    pub fn add(&mut self, b: &MatrixSym<M>) -> &mut Self{
        for i in 0..M{
            for j in 0..=i{
                self[i][j] += b[i][j];
                self[j][i] = self[i][j];
            }
        }
        self
    }

    /* Calculate the scalar product */
    pub fn scalar_prod(&mut self, b: f64) -> &mut Self{
        for i in 0..M{
            for j in 0..=i{
                self[i][j] *= b;
                self[j][i] = self[i][j];
            }
        }
        self
    }

}

/* Implementing deref to access val more easily */
impl<const M: usize> Deref for MatrixSym<M>
{
    type Target = [[f64;M];M];

    fn deref(&self) -> &Self::Target{
        &self.0.deref()
    }
}

/* Implementing derefmut to access val more easily */
impl<const M: usize> DerefMut for MatrixSym<M>
{
    fn deref_mut(&mut self) -> &mut Self::Target{
        let reference = self.0.deref_mut();
        reference
    }
}

/* Implementing Display to show results in tests */
impl<const M: usize> fmt::Display for MatrixSym<M>{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
} 

#[cfg(test)]
mod tests {
    use crate::{masy::MatrixAsym, msy::MatrixSym};

    #[test]
    fn normal_mult_t() {
        let mut a = MatrixSym::new();
        *a =   [[1.0,2.0,1.0,2.0],
                [2.0,4.0,1.0,2.0],
                [1.0,1.0,1.0,2.0],
                [2.0,2.0,2.0,2.0]];

        let mut b = MatrixAsym::new();
        *b =   [[0.0,1.0,2.0,3.0],
                [4.0,3.0,2.0,1.0]];

        let mut c = MatrixAsym::new();
        *c =   [[0.0,4.0],
                [1.0,3.0],
                [2.0,2.0],
                [3.0,1.0]];

        let d = a.b_mult_self_mult_bt(&b);

        let tmp = b.mult(&(a.0));
        let e = tmp.mult(&c);

        assert!(d.0.similar(&e, 0.001))

    }

    #[test]
    fn matrix_solve() {

        let mut a = MatrixSym::new();
        *a =   [[2.0, 1.0, 0.5],
                [1.0, 9.0, 0.1],
                [0.5, 0.1, 3.0]];

        let mut b = MatrixAsym::new();
        *b =   [[1.0],
                [1.0],
                [1.0]];

        /* Solve the equation A*x = b */
        let x = a.chol_solve(&b).unwrap();

        /* Check if A*x == b */
        let d = a.0.mult(&x);
        assert!(d.similar(&b, 0.001));
    }

    #[test]
    fn matrix_inv() {

        let mut a = MatrixSym::new();
        *a =   [[2.0, 1.0, 0.5],
                [1.0, 9.0, 0.1],
                [0.5, 0.1, 3.0]];

        let mut i = MatrixAsym::new();
        *i =   [[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]];

        let b = a.chol_solve(&i).unwrap();

        /* Check if (A⁻1)*A == I */
        assert!(b.mult(&a.0).similar(&i, 0.001));
    }

    #[test]
    fn matrix_chol() {
        /* Create a symmetric positive definite matrix */
        let mut a = MatrixSym::new();
        *a =   [[1.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 4.0]];

        /* Get the triangular matrix via Cholesky decomposition*/
        let l = a.chol().unwrap();
        print!("{} * {} = {}\n",l, l.transpose(), l.mult(&l.transpose())); 

        /* Check if A == L*(L^T) */
        assert!(a.0.similar(&l.mult(&l.transpose()), 0.001))

    }
}