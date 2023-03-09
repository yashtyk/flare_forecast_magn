import torch
import wandb
def train(train_dataloader, val_dataloader, model,  criterion,  epoch_es, epochs, optimizer, device, path):
    #wandb.init(project="resnet", entity="yshtyk")

    etalon_loss = 10000
    best_epoch = -1
    wait_epoch = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)
    for epoch in range(epochs):
        print("training epoch: {}")
        train_loss = 0

        model.train()
        for image, target in train_dataloader:
            
            image, target = image.to(device), target.to(device)

            y = model(image)
            loss = criterion(y, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().detach().cpu().item() / len(train_dataloader)
        scheduler.step()
        # validation

        with torch.no_grad():
            model.eval()
            print('validation..')
            val_loss = 0.0
            for image, target in val_dataloader:
                image, target = image.to(device), target.to(device)
                y = model(image)
                loss = criterion(y, target)
                val_loss += loss.detach().cpu().item()  / len(val_dataloader)

            print('epoch: {}, train_loss: {}, val_loss: {}, wait_epoch: {}'.format(epoch, train_loss, val_loss, wait_epoch))

            if val_loss < etalon_loss:
                wait_epoch = 0
                etalon_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), path  / 'model.pt')
                print('new etalon saved, epoch: {}, val_loss: {:.4f}, wait_epoch: {}'.format(epoch, val_loss, wait_epoch))



            elif wait_epoch <= epoch_es:
                wait_epoch +=1


            else:
                break
                print('limit of waiting epochs reached, best epoch: {}, loss: {}'.format(best_epoch, etalon_loss))

        # login loss to the wandb
       # wandb.log(
       #     {"train_loss": train_loss,
       #     "val_loss": val_loss,
       #     "waiting_epochs": wait_epoch,
       #     "best_epochs": best_epoch}
       # )

    print('training ended, best epoch: {}, val_loss: {}'.format(epoch, val_loss))









