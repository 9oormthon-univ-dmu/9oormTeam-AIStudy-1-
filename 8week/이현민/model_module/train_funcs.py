

import torch
from tqdm.auto import tqdm


def train_step(model, dataloader, loss_fn, optimizer, metric, device):
    
    # 모델을 training mode로 설정 (default state)
    model.train()
    
    # train-loss & train-accuracy for one epoch
    train_loss = 0
    train_acc  = 0
    
    for batch_idx, (X, y) in enumerate(dataloader): # X & y == a single batch
        
        X = X.to(device)
        y = y.to(device)
        
        # 1. (x 데이터를 모델에 넣고) 순방향 계산 진행 (forward pass)
        logits = model(X)

        # 2. (Batch) Training cost 계산 (Cost function 계산)
        loss = loss_fn(logits, y) # cost of batch <- nn.CrossEntropyLoss() : built-in Softmax
        train_loss += loss.item()
        
        # 3. Optimizer 내부의 이전 gradient 값 초기화 (Make "grad" to "zero")
        optimizer.zero_grad()

        # 4. Back-propagation ("Backward" propagation)
        loss.backward()

        # 5. Gradient descent 진행 (Take a "step" to update parameters)
        optimizer.step()

        # 6. (Batch) Training accuracy 계산 
        predicted_classes = logits.softmax(dim=1).argmax(dim=1)
        train_acc += metric(predicted_classes, y).item() # calculate the batch accuracy & add to the epoch accuracy

    # Batch 순회 종료 후
    train_loss = train_loss / len(dataloader) # cost of batches / num of batches (calculate average)
    train_acc  = train_acc  / len(dataloader) # acc  of batches / num of batches (calculate average)
    
    return train_loss, train_acc


def test_step(model, dataloader, loss_fn, metric, device):
    
    # 모델을 evaluation mode로 설정
    model.eval() 
    
    # test-loss & test-accuracy for one epoch
    test_loss = 0
    test_acc  = 0
    
    with torch.inference_mode(): # Set "inference mode"
        
        for batch_idx, (X, y) in enumerate(dataloader): # X & y == a single batch
            
            X = X.to(device)
            y = y.to(device)
    
            # 1. (x 데이터를 모델에 넣고) 순방향 계산 진행 (forward pass)
            logits = model(X)

            # 2. (Batch) Test cost 계산 (Cost function 계산)
            loss = loss_fn(logits, y) # cost of batch <- nn.CrossEntropyLoss() : built-in Softmax
            test_loss += loss.item()

            # 3. (Batch) Test accuracy 계산 
            predicted_classes = logits.softmax(dim=1).argmax(dim=1)
            test_acc += metric(predicted_classes, y).item() # calculate the batch accuracy & add to the epoch accuracy

    
    # Batch 순회 종료 후
    test_loss = test_loss / len(dataloader) # cost of batches / num of batches (calculate average)
    test_acc  = test_acc  / len(dataloader) # acc  of batches / num of batches (calculate average)
    
    return test_loss, test_acc


def train(model, 
          train_dataloader, 
          test_dataloader, 
          optimizer, 
          loss_fn, 
          metric, 
          device, 
          epochs):
    
    results = {"train_loss": [], 
               "train_acc" : [], 
               "test_loss" : [], 
               "test_acc"  : []}
    
    for epoch in tqdm(range(epochs)): # from tqdm.auto import tqdm
        
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn, 
                                           optimizer=optimizer, 
                                           metric=metric, 
                                           device=device)
        
        test_loss, test_acc   = test_step(model=model,
                                          dataloader=test_dataloader, 
                                          loss_fn=loss_fn, 
                                          metric=metric, 
                                          device=device)
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        print('Epoch : {} | Train_loss : {} | Train_acc : {} | Test_loss : {} | Test_acc : {}'.format(epoch+1, 
                                                                                                      train_loss, 
                                                                                                      train_acc, 
                                                                                                      test_loss, 
                                                                                                      test_acc))
    return results
