//
//  ViewController.h
//  XOR_Example
//
//  Created by Kurt Jacobs on 2015/06/12.
//  Copyright (c) 2015 RandomDudes. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "Torch.h"
#include <Torch/Torch.h>
#import "XORClassifyObject.h"

@interface ViewController : UIViewController

@property (nonatomic, strong) Torch *t;
@property (weak, nonatomic) IBOutlet UILabel *answerLabel;
@property (weak, nonatomic) IBOutlet UITextField *valueOneTextfield;
@property (weak, nonatomic) IBOutlet UITextField *valueTwoTextField;

@end

