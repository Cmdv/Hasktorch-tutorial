{-# LANGUAGE ExtendedDefaultRules #-}

module Main where

import Torch.Autograd (makeIndependent)
import Torch (IndependentTensor(..), grad)
import Torch.Random (mkGenerator, randn')
import Torch.Device (Device(..), DeviceType(..))
import Torch.NN (Linear(..), LinearSpec(..), HasForward(..), sample)
import qualified Torch.Functional as TF
import Torch.TensorFactories (zeros')
import Torch.Tensor ( TensorLike(..)
                    , select , toDouble
                    , shape , dtype
                    , (!) , toInt
                    , size , dim
                    , numel, slice)
import Torch.Functional.Internal (vdot)

{-
 ==================================================================
                         Table of Contents
 ==================================================================

 1. Basic Tensor                           (Line 35  to 89)
 1. Basic autograd example 1               (Line 31  to 47)
 2. Basic autograd example 2               (Line 49  to __)
 3. Loading data from numpy                (Line __  to __)
 4. Input pipline                          (Line __  to __)
 5. Input pipline for custom dataset       (Line __  to __)
 6. Pretrained model                       (Line __  to __)
 7. Save and load model                    (Line __  to __)

 ==================================================================
                     1. Basic Tensor
 ==================================================================
-}

tensorsBasic :: IO ()
tensorsBasic = do
  let tensor1  = asTensor (1 :: Int)
      tensor2  = asTensor ([1, 2, 3, 4] :: [Int])
      tensor3  = asTensor ([[1, 2, 3, 4], [5,6,7,8]] :: [[Int]])
      tensor4  = asTensor ([1.0, 2.0, 3.0] :: [Float])
  printTensor "1-Tensor1: " tensor1
  -- Tensor Int64 []  1
  printTensor "1-Tensor2: " tensor2
  -- Tensor Int64 [4] [ 1,  2,  3,  4]
  printTensor "1-select 0 0: " $ select 0 0 tensor2
  -- Tensor Int64 [] 1
  printTensor "1-select 0 1: " $ select 0 1 tensor3
  -- Tensor Int64 [4] [ 5,  6,  7,  8]
  printTensor "1-select 1 2: " $ select 1 2 tensor3
  -- Tensor Int64 [2] [ 3,  7]
  printTensor "1-size: " $ size 0 tensor2
  -- 4
  printTensor "1-dim: " $ dim tensor2
  -- 1
  printTensor "1-shape: " $ shape tensor2
  -- [4]
  printTensor "1-shape: " $ shape tensor3
  -- [2, 4]
  printTensor "1-numel: " $ numel tensor2
  -- 8
  printTensor "1-toDouble: " $ toDouble $ select 0 0 tensor2
  -- 1.0
  printTensor "1-toInt: " $ toInt $ select 0 2 tensor4
  -- 3
  printTensor "1-asValue Int: " (asValue tensor1 :: Int)
  -- 1
  printTensor "1-asValue [Int]: " (asValue tensor2 :: [Int])
  -- [1,2,3,4]
  printTensor "1-dtype: " $ dtype tensor1
  -- Int64
  printTensor "1-dtype: " $ dtype tensor4
  -- Float
  printTensor "1-view: " $ TF.view [4,1] tensor2
  -- change the shape of the tensor
  -- Tensor Int64 [4,1] [[ 1], [ 2], [ 3], [ 4]]
  printTensor "1-view -1: " $ TF.view [-1,1] tensor2
  -- (-1) will infer the number of rows in the new tensor for us
  -- Tensor Int64 [4,1] [[ 1], [ 2], [ 3], [ 4]]
  printTensor "1-slice 1D: " $ slice 0 1 3 1 tensor2
  -- Tensor Int64 [2] [ 2,  3]
  printTensor "1-add: " $ tensor2 + tensor2
  -- Tensor Int64 [4] [ 2,  4,  6,  8]
  printTensor "1-multiplication: " $ tensor2 * tensor2
  -- Tensor Int64 [4] [ 1,  4,  9,  16]
  printTensor "1-multiplication with scaler: " $ tensor2 * 2
  -- Tensor Int64 [4] [ 2,  4,  6,  8]
  printTensor "1-dot product: " $ TF.dot tensor2 tensor2
  -- Tensor Int64 []  30
  printTensor "1-mean: " $ TF.mean tensor4
  -- Tensor Float []  2.0000
  printTensor "1-max: " $ TF.max tensor2


{-
 ==================================================================
                     1. Two Dimensional Tensors
 ==================================================================
-}


tensorsN :: IO ()
tensorsN = do
  let tensor2D = asTensor ([[1, 2, 3, 4], [5,6,7,8]] :: [[Int]])
      tensor3D = asTensor ([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]]] :: [[[Int]]])
  printTensor "1-select 0 1: " $ select 0 1 tensor2D
  -- Tensor Int64 [4] [ 5,  6,  7,  8]
  printTensor "1-select 1 2: " $ select 1 2 tensor2D
  -- Tensor Int64 [2] [ 3,  7]
  printTensor "1-size: " $ size 0 tensor2D
  -- 4
  printTensor "1-dim: " $ dim tensor2D
  -- 1
  printTensor "1-dim: " $ dim tensor3D
  -- 3
  printTensor "1-shape: " $ shape tensor2D
  -- [2, 4]
  printTensor "1-numel: " $ numel tensor2D
  -- 8
  printTensor "1-view: " $ TF.view [8,1] tensor2D
  -- change the shape of the tensor
  -- Tensor Int64 [8,1] [[ 1], [ 2], [ 3], [ 4], [ 5], [ 6], [ 7], [ 8]]
  printTensor "1-view -1: " $ TF.view [-1,1] tensor2D
  -- (-1) will infer the number of rows in the new tensor for us
  -- Tensor Int64 [8,1] [[ 1], [ 2], [ 3], [ 4], [ 5], [ 6], [ 7], [ 8]]
  printTensor "1-slice 2D: " $ slice 1 1 3 1 tensor2D
  -- Tensor Int64 [2,2] [[ 2,  3], [ 6,  7]]
  let r = tensor3D ! (1,0)
  -- (1,0) is ambiguous type variable ExtendedDefaultRules is used
  printTensor "1-tensor4 ! (1,0): " r
  -- Tensor Int64 [3] [ 6,  7,  8]
  printTensor "1-add: " $ tensor2D + tensor2D
  -- Tensor Int64 [4] [ 2,  4,  6,  8]
  printTensor "1-multiplication: " $ tensor2D * tensor2D
  -- Tensor Int64 [4] [ 1,  4,  9,  16]
  printTensor "1-multiplication with scaler: " $ tensor2D * 2
  -- Tensor Int64 [4] [ 2,  4,  6,  8]
  -- printTensor "1-dot product: " $ vdot tensor2D tensor2D
  -- Tensor Int64 []  30
  -- printTensor "1-mean: " $ TF.mean tensor2D
  -- Tensor Float []  2.0000
  -- printTensor "1-max: " $ TF.max tensor2D


  -- add 2 tensors
  -- vector multiplication with scaler 2 * [1,2] == [2,4]
  -- (Hadamard) product of 2 tensors [2,4] * [2,4] = [4,16]
  -- dot product [2,4] [2,4] == (2 * 2) + (4 * 4) = 20
  -- mean [1, -1, 1, -1] == 1/4 (1-1 + 1-1) == 0
  -- max [1,2,-1,5] == 5
  -- linespace



{-
 ==================================================================
                     1. Basic autograd example 1
 ==================================================================
-}


twoBasicAutograd :: IO ()
twoBasicAutograd = do
  x <- makeIndependent $ asTensor (1.0 :: Float)
  w <- makeIndependent $ asTensor (2.0 :: Float)
  b <- makeIndependent $ asTensor (3.0 :: Float)
  let y = toDependent w * toDependent x + toDependent b
      gradients = grad y [x, w, b]
  printTensor "2 - IndependentTensor: " x
  -- IndependentTensor {toDependent = Tensor Float []  1.0000   }
  printTensor "2 - IndependentTensor: " w
  -- IndependentTensor {toDependent = Tensor Float []  2.0000   }
  printTensor "2 - IndependentTensor: " b
  -- IndependentTensor {toDependent = Tensor Float []  3.0000   }
  printTensor "2 - Gradients: " gradients
  -- [Tensor Float []  2.0000   ,Tensor Float []  1.0000   ,Tensor Float []  1.0000   ]
  printTensor "2 - Calculation: " y
  -- Tensor Float []  5.0000

{-
 ==================================================================
                     2. Basic autograd example 2
 ==================================================================
-}


threeBasicAutograd :: IO ()
threeBasicAutograd = do
  --  Create random tensors of the shape (10, 3) and (10, 2)
  generator <- mkGenerator (Device CPU 0) 0
  let (x, next) = randn' [10, 3] generator
      (y, _)    = randn' [10, 2] next
  printTensor "3 - Random (10, 3): " x
  printTensor "3 - Random (10, 2): " y

  -- Build a fully connected layer
  linear <- sample $ LinearSpec { in_features = 3, out_features = 2 }
  printTensor "3 - Weight: " (weight linear)
  printTensor "3 - Bias: " (bias linear)

  -- forward pass
  let prediction = forward linear x
  -- compute loss
      loss = TF.mseLoss prediction y

  printTensor "3 - Prediction: " prediction
  printTensor "3 - Prediction: " prediction
  printTensor "3 - Loss: " loss
  printTensor "3 - Zeros: " (zeros' [2, 3])

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


printTensor :: Show a => String -> a -> IO()
printTensor s t = do
  putStr $ s ++ "\n" ++ show t ++ "\n\n"

main :: IO ()
main = do
  -- tensorsBasic
  tensorsN
  --twoBasicAutograd
  --threeBasicAutograd
