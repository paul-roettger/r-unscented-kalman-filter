use std::ops::{DerefMut, Deref};
use std::fmt;

use crate::matrix_asym as masy;


#[derive(Copy, Clone)]
pub struct MatrixSym<const M: usize>(pub masy::MatrixAsym<M,M>);


impl<const M: usize> MatrixSym<M>{
    pub fn new() -> Self{
        Self { 0: masy::MatrixAsym::new() }
    }

    pub fn b_mult_self_mult_bt<const U: usize>(&self, b: &masy::MatrixAsym<M,U>) -> MatrixSym<U>{
        let mut result = MatrixSym::new();
        let mut tmp = [[0.0; M]; U];
        let mut sum;
        for i in 0..U{
            for j in 0..M {
                sum = 0.0;
                for k in 0..M {
                    sum += b[i][k] * self[k][j];
                }
                tmp[i][j] = sum;
            }
        }
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

    pub fn chol(&self) -> Result<masy::MatrixAsym<M,M>, &'static str>{
        let mut sum;
        let mut result = masy::MatrixAsym::new();

        /* Cholesky decomposition */
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

    pub fn chol_solve<const U: usize>(&self, b: &masy::MatrixAsym<U,M>) -> Result<masy::MatrixAsym<U,M>, &'static str>{
        let mut result = masy::MatrixAsym::new();
        let mut m_y = [[0.0;U];M];
        let mut sum;

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

    pub fn sub(&mut self, b: &MatrixSym<M>) -> &mut Self{
        for i in 0..M{
            for j in 0..=i{
                self[i][j] -= b[i][j];
                self[j][i] = self[i][j];
            }
        }
        self
    }

    pub fn add(&mut self, b: &MatrixSym<M>) -> &mut Self{
        for i in 0..M{
            for j in 0..=i{
                self[i][j] += b[i][j];
                self[j][i] = self[i][j];
            }
        }
        self
    }

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

impl<const M: usize> Deref for MatrixSym<M>
{
    type Target = [[f64;M];M];

    fn deref(&self) -> &Self::Target{
        &self.0.deref()
    }
}

impl<const M: usize> DerefMut for MatrixSym<M>
{
    fn deref_mut(&mut self) -> &mut Self::Target{
        let reference = self.0.deref_mut();
        reference
    }
}

impl<const M: usize> fmt::Display for MatrixSym<M>{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
} 
