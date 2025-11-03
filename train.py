import torch
from tqdm import tqdm
import ot 

def train_one_epoch_wasserstein(model, train_loader, optimizer, loss_fn, hooks, save_output, device, args):

    model.train()
    running_loss=0
    running_cross_loss=0
    running_wasserstein_cost = 0
    running_wasserstein_cost_each_layer = torch.zeros(len(hooks.keys())-1)
    correct=0
    total=0
    for data in tqdm(train_loader):

        inputs,labels=data[0].to(device),data[1].to(device)
        
        outputs=model(inputs)
        wasserstein_cost = torch.zeros(int(len(hooks.keys())/2)).to(device)
        i=0
        for k in range(0,len(hooks.keys())-1,2):
            wasserstein_cost[i] = ot.sliced.max_sliced_wasserstein_distance(X_s=save_output.outputs[k].reshape(args.batch_size,-1), X_t=save_output.outputs[k+1].reshape(args.batch_size,-1), seed=None)
            i+=1

        save_output.clear()
        mean_wasserstein_cost = torch.mean(wasserstein_cost)

        cross_loss =loss_fn(outputs,labels)
        loss = cross_loss + args.lambda_reg*mean_wasserstein_cost

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_cross_loss += cross_loss.item()
        running_wasserstein_cost += mean_wasserstein_cost.item()
        for l in range(len(wasserstein_cost)):
            running_wasserstein_cost_each_layer[l]+= wasserstein_cost[l].item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    
    train_loss=running_loss/len(train_loader)
    cross_loss=running_cross_loss/len(train_loader)
    mean_wasserstein_cost = running_wasserstein_cost/len(train_loader)
    wasserstein_cost_each_layer = running_wasserstein_cost_each_layer/len(train_loader)
    train_accu=100.*correct/total

    return train_accu, train_loss, cross_loss, mean_wasserstein_cost, wasserstein_cost_each_layer

def val_one_epoch_wasserstein(model, val_loader, loss_fn, hooks, save_output, device, args):

    model.eval()
    running_loss=0
    running_cross_loss=0
    running_wasserstein_cost = 0
    running_wasserstein_cost_each_layer = torch.zeros(len(hooks.keys())-1)
    correct=0
    total=0

    with torch.no_grad():
        
        for data in tqdm(val_loader):
            
            inputs,labels=data[0].to(device),data[1].to(device)
            
            outputs=model(inputs)
            wasserstein_cost = torch.zeros(int(len(hooks.keys())/2))
            i=0
            for k in range(0,len(hooks.keys())-1,2):
                wasserstein_cost[i] = ot.sliced.sliced_wasserstein_distance(X_s=save_output.outputs[k].reshape(args.batch_size,-1), X_t=save_output.outputs[k+1].reshape(args.batch_size,-1), seed=args.seed)
                i+=1
            

            save_output.clear()
            mean_wasserstein_cost = torch.mean(wasserstein_cost)

            cross_loss =loss_fn(outputs,labels)
            loss = cross_loss + args.lambda_reg*mean_wasserstein_cost
            
            running_loss += loss.item()
            running_cross_loss += cross_loss.item()
            running_wasserstein_cost += mean_wasserstein_cost.item()
            for l in range(len(wasserstein_cost)):
                running_wasserstein_cost_each_layer[l]+= wasserstein_cost[l].item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
    val_loss=running_loss/len(val_loader)
    cross_loss=running_cross_loss/len(val_loader)
    mean_wasserstein_cost = running_wasserstein_cost/len(val_loader)
    wasserstein_cost_each_layer = running_wasserstein_cost_each_layer/len(val_loader)
    val_accu=100.*correct/total

    return val_accu, val_loss, cross_loss, mean_wasserstein_cost, wasserstein_cost_each_layer

def train_one_epoch_wasserstein_2(model, train_loader, optimizer, loss_fn, hooks, save_input, save_output, device, args):

    model.train()
    running_loss=0
    running_cross_loss=0
    running_wasserstein_cost = 0
    running_wasserstein_cost_each_layer = torch.zeros(len(hooks.keys())-1)
    correct=0
    total=0
    for data in tqdm(train_loader):

        inputs,labels=data[0].to(device),data[1].to(device)
        
        outputs=model(inputs)
        wasserstein_cost = torch.zeros(int(len(hooks.keys())/2)).to(device)

        for k in range(0,int(len(hooks.keys())/2)):
            wasserstein_cost[k] = ot.sliced.max_sliced_wasserstein_distance(X_s=save_output.outputs[k].reshape(args.batch_size,-1), X_t=save_input.inputs[k][0].reshape(args.batch_size,-1), seed=None)

        save_input.clear()
        save_output.clear()
        mean_wasserstein_cost = torch.mean(wasserstein_cost)

        cross_loss =loss_fn(outputs,labels)
        loss = cross_loss + args.lambda_reg*mean_wasserstein_cost

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_cross_loss += cross_loss.item()
        running_wasserstein_cost += mean_wasserstein_cost.item()
        for l in range(len(wasserstein_cost)):
            running_wasserstein_cost_each_layer[l]+= wasserstein_cost[l].item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    
    train_loss=running_loss/len(train_loader)
    cross_loss=running_cross_loss/len(train_loader)
    mean_wasserstein_cost = running_wasserstein_cost/len(train_loader)
    wasserstein_cost_each_layer = running_wasserstein_cost_each_layer/len(train_loader)
    train_accu=100.*correct/total

    return train_accu, train_loss, cross_loss, mean_wasserstein_cost, wasserstein_cost_each_layer

def val_one_epoch_wasserstein_2(model, val_loader, loss_fn, hooks, save_input, save_output, device, args):
    
    model.eval()
    running_loss=0
    running_cross_loss=0
    running_wasserstein_cost = 0
    running_wasserstein_cost_each_layer = torch.zeros(len(hooks.keys())-1)
    correct=0
    total=0

    with torch.no_grad():
        
        for data in tqdm(val_loader):
            
            inputs,labels=data[0].to(device),data[1].to(device)
            
            outputs=model(inputs)
            wasserstein_cost = torch.zeros(int(len(hooks.keys())/2))
            for k in range(0,int(len(hooks.keys())/2)):
                wasserstein_cost[k] = ot.sliced.max_sliced_wasserstein_distance(X_s=save_output.outputs[k].reshape(args.batch_size,-1), X_t=save_input.inputs[k][0].reshape(args.batch_size,-1), seed=None)
            

            save_output.clear()
            save_input.clear()
            mean_wasserstein_cost = torch.mean(wasserstein_cost)

            cross_loss =loss_fn(outputs,labels)
            loss = cross_loss + args.lambda_reg*mean_wasserstein_cost
            
            running_loss += loss.item()
            running_cross_loss += cross_loss.item()
            running_wasserstein_cost += mean_wasserstein_cost.item()
            for l in range(len(wasserstein_cost)):
                running_wasserstein_cost_each_layer[l]+= wasserstein_cost[l].item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
    val_loss=running_loss/len(val_loader)
    cross_loss=running_cross_loss/len(val_loader)
    mean_wasserstein_cost = running_wasserstein_cost/len(val_loader)
    wasserstein_cost_each_layer = running_wasserstein_cost_each_layer/len(val_loader)
    val_accu=100.*correct/total

    return val_accu, val_loss, cross_loss, mean_wasserstein_cost, wasserstein_cost_each_layer

def val_one_epoch(model, val_loader, loss_fn, device, args):

    model.eval()
    running_loss=0
    correct=0
    total=0

    with torch.no_grad():
        
        for data in tqdm(val_loader):
            
            inputs,labels=data[0].to(device),data[1].to(device)
            
            outputs=model(inputs)

            loss =loss_fn(outputs,labels)
            
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            if args.dataset == "Flowers-102" or args.dataset == "DTD":
                correct += predicted.eq(torch.argmax(labels, dim=1)).sum().item()
            else:
                correct += predicted.eq(labels).sum().item()
        
    val_loss=running_loss/len(val_loader)
    val_accu=100.*correct/total

    return val_accu, val_loss