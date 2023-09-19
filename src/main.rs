use std::ops::{DerefMut, Deref};
use std::fmt;
use std::convert::From;
use futures::executor::block_on;

fn main() {
    println!("Hello, world!");
}

struct MatrixAsym<const M: usize, const N: usize>{
    pub val: [[f64;M];N]
}

struct MatrixSym<const M: usize>(MatrixAsym<M,M>);

impl<const M: usize, const N: usize> MatrixAsym<M,N>{

    pub fn new() -> Self{
        Self { val: [[0.0; M];N] }
    }

    pub async fn add(&mut self, b: &MatrixAsym<M,N>) -> &Self{
        for (linea, lineb) in self.deref_mut().iter_mut().zip(b.deref().iter()){
            for (a,b) in linea.iter_mut().zip(lineb.iter()){
                *a += *b;
            }
        }
        self
    }

    pub async fn mult<const U: usize>(&self, b: &MatrixAsym<U,M>) -> MatrixAsym<U,N>{

        let mut result = MatrixAsym::new();
        let mut sum;
        for i in 0..N {
            for j in 0..U {
                sum = 0.0;
                for k in 0..M {
                    sum += self.val[i][k] * b.val[k][j];
                }
                result.val[i][j] = sum;
            }
        }
        result
    }
}

impl<const M: usize, const N: usize> Deref for MatrixAsym<M,N>
{
    type Target = [[f64;M];N];

    fn deref(&self) -> &Self::Target{
        &self.val
    }
}

impl<const M: usize, const N: usize> DerefMut for MatrixAsym<M,N>
{
    fn deref_mut(&mut self) -> &mut Self::Target{
        &mut self.val
    }
}


impl<const M: usize, const N: usize> fmt::Display for MatrixAsym<M,N>{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[\n")?;
        for line in self.deref(){
            write!(f, "[")?;
            for item in line{
                write!(f, "{},", *item)?;
            }
            write!(f, "]\n")?;
        }
        write!(f, "]")
    }
}


impl<const M: usize> MatrixSym<M>{
    pub fn new() -> Self{
        Self { 0: MatrixAsym::new() }
    }

    pub async fn b_mul_self_mult_bt<const U: usize>(&self, b: &MatrixAsym<M,U>) -> MatrixSym<U>{
        let mut result = MatrixSym::new();
        let mut tmp = [[0.0; M]; U];
        let mut sum;
        for i in 0..U{
            for j in 0..M {
                sum = 0.0;
                for k in 0..M {
                    sum += b.val[i][k] * self.0.val[k][j];
                }
                tmp[i][j] = sum;
            }
        }
        for i in 0..U {
            for j in 0..=i {
                sum = 0.0;
                for k in 0..M {
                    sum += tmp[i][k] * b.val[j][k];
                }
                result.0.val[i][j] = sum;
                result.0.val[j][i] = sum;
            }
        }
        result
    }

    pub fn symmetrize(&mut self){
        for i in 1..M{
            for j in 0..i{
                self.0.val[j][i] += self.0.val[i][j];
                self.0.val[j][i] /= 2.0;
                self.0.val[i][j] = self.0.val[j][i];
            }
        }
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

#[cfg(test)]
mod tests {

    use futures::executor::block_on;

    use crate::{MatrixAsym, MatrixSym};

    #[test]
    fn add_test() {
        let mut a = MatrixAsym::new();
        *a = [[1.0,2.0],
              [3.0,4.0],
              [5.0,6.0],
              [7.0,8.0]];

        let mut b = MatrixAsym::new();
        *b = [[-1.0,-2.0],
              [-3.0,-4.0],
              [-5.0,-6.0],
              [-7.0,-8.0]];

        let c = block_on(a.add(&b));
        print!("{}",*c);     
 
    }

    #[test]
    fn mul_test() {
        let mut a = MatrixAsym::new();
        *a = [[3.0,2.0,1.0],
              [1.0,0.0,2.0],
              [3.0,2.0,1.0],
              [1.0,0.0,2.0]];

        let mut b = MatrixAsym::new();
        *b = [[1.0,2.0],
              [0.0,1.0],
              [4.0,0.0]];

        let c = block_on(a.mult(&b));

        print!("{}\n*{}\n={}",a,b,c);     
 
    }    

    #[test]
    fn normal_test() {
        let mut a = MatrixSym::new();
        *a =   [[1.0,2.0,1.0,2.0],
                [3.0,4.0,1.0,2.0],
                [5.0,6.0,1.0,2.0],
                [7.0,8.0,1.0,2.0]];

        a.symmetrize();
        print!("{}\n",a);    
 
    }

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

        let d = block_on(a.b_mul_self_mult_bt(&b));
        print!("{}\n",d); 

        let tmp = block_on(b.mult(&(a.0)));
        let e = block_on(tmp.mult(&c));
        print!("{}\n",e); 
 
    }

}