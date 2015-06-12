//
//  ViewController.m
//  XOR_Example
//
//  Created by Kurt Jacobs on 2015/06/12.
//  Copyright (c) 2015 RandomDudes. All rights reserved.
//

#import "ViewController.h"
#include <Torch/Torch.h>
#import "Torch.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
  [super viewDidLoad];
  Torch *t = [Torch new];
  [t initialize];
  [t runMain:@"main" inFolder:@"xor_lua"];
}

- (void)didReceiveMemoryWarning
{
  [super didReceiveMemoryWarning];
}

@end
