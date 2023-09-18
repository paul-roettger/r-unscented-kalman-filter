use std::ops::{DerefMut, Deref};
use std::fmt;
use std::convert::From;
use futures::executor::block_on;

fn main() {
    println!("Hello, world!");
}


struct MatrixAsym{
    pub m: usize,
    pub n: usize,
    pub val: Box<[f64]>
}

struct MatrixSym{
    pub m: usize,
    pub val: Box<[f64]>
}

impl MatrixAsym {
    pub async fn new(m: usize, n: usize) -> Self{

        Self { m: m, 
               n: n, 
               val: vec![0.0; m*n].into_boxed_slice() }
    }

    pub async fn add(&mut self, b: &MatrixAsym) -> Result<&Self, &'static str>{
        if self.m != b.m || self.n != b.n{
            Err("Invalid dimension")
        }else{
            for (a, b) in self.val.deref_mut().iter_mut().zip(b.val.iter()){
                *a += b;
            }
            Ok(self)
        }
    }

    pub async fn mult(&self, b: &MatrixAsym) -> Result<Self, &'static str>{
        if self.m != b.n{
            Err("Invalid dimension")
        }else{
            let mut result = MatrixAsym::new(self.n, b.m).await;
            let mut i: usize;
            let mut j: usize;
            for (i_loop, c) in result.val.deref_mut().iter_mut().enumerate(){
                i = i_loop % result.m;
                j = i_loop/result.m;
                *c = self.val.deref().iter().skip(j*result.m).take(result.m)
                    .zip(b.val.deref().iter().skip(i).step_by(b.m))
                    .map(|(a,b)| *a*(*b)).sum();
            }
            Ok(result)
        }
    }
}

impl fmt::Display for MatrixAsym{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let array = &self.val;

        write!(f, "[")?;

        for (i, item) in array.iter().enumerate() {
            if i % self.m == 0{
                write!(f, "\n ")?;
            } else if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", *item)?;
        }

        write!(f, "]")
    }
}


impl MatrixSym {
    pub async fn new(m: usize) -> Self{
        Self { m: m, 
               val: vec![0.0; m*m].into_boxed_slice() }
    }

    pub async fn b_mul_self_mult_bt(&self, b: &MatrixAsym) -> Result<MatrixSym, &'static str>{
        if self.m != b.n{
            Err("Invalid dimension")
        }else{
            let mut result = MatrixSym::new(self.m).await;
            let mut i: usize;
            let mut j: usize;
            for (i_loop, c) in result.val.deref_mut().iter_mut().enumerate(){
                i = i_loop % result.m;
                j = i_loop/result.m;
                if i>j {
                    (i,j) = (j,i);
                }
                *c = self.val.deref().iter().skip(j*result.m).take(result.m)
                    .zip(b.val.deref().iter().skip(i).step_by(b.m))
                    .map(|(a,b)| *a*(*b)).sum();
            }
            Ok(result)
        }
    }

    pub async fn normalize(&mut self){
        let mut i: usize;
        let mut j: usize;
        let mut i_transp: usize;
        let mut var : Vec<_> = self.val.deref_mut().iter_mut().collect();
        for i_loop in 0..var.len(){
            i = i_loop % self.m;
            j = i_loop/self.m;
            if i < j{
                i_transp = i*self.m+j;
                *var[i_loop] += *var[i_transp];
                *var[i_loop] /= 2.0;
                *var[i_transp] = *var[i_loop]
            }
        }
    }
}

impl fmt::Display for MatrixSym{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let array = &self.val;

        write!(f, "[")?;

        for (i, item) in array.iter().enumerate() {
            if i % self.m == 0{
                write!(f, "\n ")?;
            } else if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", *item)?;
        }

        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {

    use futures::executor::block_on;

    use crate::{MatrixAsym, MatrixSym};

    #[test]
    fn add_test() {
        let mut a = MatrixAsym{
            m: 2,
            n: 4,
            val: vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0].into_boxed_slice()
        };

        let b = MatrixAsym{
            m: 2,
            n: 4,
            val: vec![-1.0,-2.0,-3.0,-4.0,-5.0,-6.0,-7.0,-8.0].into_boxed_slice()
        };
        let c = block_on(a.add(&b)).unwrap();

        print!("{}",*c);     
 
    }

    #[test]
    fn mul_test() {
        let a = MatrixAsym{
            m: 3,
            n: 3,
            val: vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0].into_boxed_slice()
        };

        let b = MatrixAsym{
            m: 3,
            n: 3,
            val: vec![1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0].into_boxed_slice()
        };
        let c = block_on(a.mult(&b)).unwrap();

        print!("{}\n*{}\n={}",a,b,c);     
 
    }

    #[test]
    fn normal_test() {
        let mut a = MatrixSym{
            m: 3,
            val: vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0].into_boxed_slice()
        };
        print!("{}\n",a); 

        block_on(a.normalize());
        print!("{}\n",a);    
 
    }

}