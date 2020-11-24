module Main where

import Torch.Autograd (makeIndependent)
import Torch.Tensor (TensorLike(..))
import Torch (IndependentTensor(..))
import Torch (grad)

{-
 ==================================================================
                         Table of Contents
 ==================================================================

 1. Basic autograd example 1               (Line 28  to 40)
 2. Basic autograd example 2               (Line 49  to __)
 3. Loading data from numpy                (Line __  to __)
 4. Input pipline                          (Line __  to __)
 5. Input pipline for custom dataset       (Line __  to __)
 6. Pretrained model                       (Line __  to __)
 7. Save and load model                    (Line __  to __)


 ==================================================================
                     1. Basic autograd example 1
 ==================================================================
-}


basicAutograd1 :: IO ()
basicAutograd1 = do
  x <- makeIndependent $ asTensor (1.0 :: Float)
  w <- makeIndependent $ asTensor (2.0 :: Float)
  b <- makeIndependent $ asTensor (3.0 :: Float)
  let y = toDependent w * toDependent x + toDependent b
      gradients = grad y [x, w, b]
  putStrLn $ show  x        -- IndependentTensor {toDependent = Tensor Float []  1.0000   }
  putStrLn $ show  w        -- IndependentTensor {toDependent = Tensor Float []  2.0000   }
  putStrLn $ show  b        -- IndependentTensor {toDependent = Tensor Float []  3.0000   }
  putStrLn $ show gradients -- [Tensor Float []  2.0000   ,Tensor Float []  1.0000   ,Tensor Float []  1.0000   ]
  putStrLn $ show y         -- Tensor Float []  5.0000
  return ()

{-
 ==================================================================
                     2. Basic autograd example 2
 ==================================================================
-}


basicAutograd2 :: IO ()
basicAutograd2 = undefined


-- # Create tensors of shape (10, 3) and (10, 2).
-- x = torch.randn(10, 3)
-- y = torch.randn(10, 2)

-- # Build a fully connected layer.
-- linear = nn.Linear(3, 2)
-- print ('w: ', linear.weight)
-- print ('b: ', linear.bias)

-- # Build loss function and optimizer.
-- criterion = nn.MSELoss()
-- optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

-- # Forward pass.
-- pred = linear(x)

-- # Compute loss.
-- loss = criterion(pred, y)
-- print('loss: ', loss.item())

-- # Backward pass.
-- loss.backward()

-- # Print out the gradients.
-- print ('dL/dw: ', linear.weight.grad)
-- print ('dL/db: ', linear.bias.grad)

-- # 1-step gradient descent.
-- optimizer.step()

-- # You can also perform gradient descent at the low level.
-- # linear.weight.data.sub_(0.01 * linear.weight.grad.data)
-- # linear.bias.data.sub_(0.01 * linear.bias.grad.data)

-- # Print out the loss after 1-step gradient descent.
-- pred = linear(x)
-- loss = criterion(pred, y)
-- print('loss after 1 step optimization: ', loss.item())



main :: IO ()
main = do
  basicAutograd1
