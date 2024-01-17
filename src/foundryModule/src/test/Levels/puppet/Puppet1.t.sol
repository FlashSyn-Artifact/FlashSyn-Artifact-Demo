// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

import {DSTest} from "ds-test/test.sol";
import {Utilities} from "../../utils/Utilities.sol";
import {console} from "../../utils/Console.sol";
import {Vm} from "forge-std/Vm.sol";
import {stdCheats} from "forge-std/stdlib.sol";

import {DamnValuableToken} from "../../../Contracts/DamnValuableToken.sol";
import {PuppetPool} from "../../../Contracts/puppet/PuppetPool.sol";

import {Strings} from "openzeppelin-contracts/utils/Strings.sol";

import "ds-test/test.sol";




interface UniswapV1Exchange {
    function addLiquidity(
        uint256 min_liquidity,
        uint256 max_tokens,
        uint256 deadline
    ) external payable returns (uint256);
    function balanceOf(address _owner) external view returns (uint256);
    function getTokenToEthInputPrice(uint256 tokens_sold)
        external
        view
        returns (uint256);
    function tokenToEthSwapInput(uint256 tokens_sold, uint256 min_eth, uint256 deadline) external returns (uint256);
    function ethToTokenSwapInput(uint256 min_tokens, uint256 deadline)  external payable returns (uint256);
}

interface UniswapV1Factory {
    function initializeFactory(address template) external;
    function createExchange(address token) external returns (address);
}

contract PuppetStep1 is DSTest, stdCheats {
    Vm internal immutable vm = Vm(HEVM_ADDRESS);

    // Uniswap exchange will start with 10 DVT and 10 ETH in liquidity
    uint256 internal constant UNISWAP_INITIAL_TOKEN_RESERVE = 10e18;
    uint256 internal constant UNISWAP_INITIAL_ETH_RESERVE = 10e18;

    uint256 internal constant ATTACKER_INITIAL_TOKEN_BALANCE = 1_000e18;
    uint256 internal constant ATTACKER_INITIAL_ETH_BALANCE = 25e18;
    uint256 internal constant POOL_INITIAL_TOKEN_BALANCE = 100_000e18;
    uint256 internal constant DEADLINE = 10_000_000;

    UniswapV1Exchange internal uniswapV1ExchangeTemplate;
    UniswapV1Exchange internal uniswapExchange;
    UniswapV1Factory internal uniswapV1Factory;

    DamnValuableToken internal dvt;
    PuppetPool internal puppetPool;
    address payable internal attacker;

    uint256 DVTasked;
    uint256 ethNeed;

    uint DVTspent;
    uint ethGot;

    string str89;
    string str90;
    string str91;



    function setUp() public {
        /** SETUP SCENARIO - NO NEED TO CHANGE ANYTHING HERE */
        attacker = payable(
            address(uint160(uint256(keccak256(abi.encodePacked("attacker")))))
        );
        vm.label(attacker, "Attacker");
        vm.deal(attacker, ATTACKER_INITIAL_ETH_BALANCE);

        // Deploy token to be traded in Uniswap
        dvt = new DamnValuableToken();
        vm.label(address(dvt), "DVT");

        uniswapV1Factory = UniswapV1Factory(
            deployCode("./src/build-uniswap/v1/UniswapV1Factory.json")
        );

        // Deploy a exchange that will be used as the factory template
        uniswapV1ExchangeTemplate = UniswapV1Exchange(
            deployCode("./src/build-uniswap/v1/UniswapV1Exchange.json")
        );

        // Deploy factory, initializing it with the address of the template exchange
        uniswapV1Factory.initializeFactory(address(uniswapV1ExchangeTemplate));

        uniswapExchange = UniswapV1Exchange(
            uniswapV1Factory.createExchange(address(dvt))
        );

        vm.label(address(uniswapExchange), "Uniswap Exchange");

        // Deploy the lending pool
        puppetPool = new PuppetPool(address(dvt), address(uniswapExchange));
        vm.label(address(puppetPool), "Puppet Pool");

        // Add initial token and ETH liquidity to the pool
        dvt.approve(address(uniswapExchange), UNISWAP_INITIAL_TOKEN_RESERVE);
        uniswapExchange.addLiquidity{value: UNISWAP_INITIAL_ETH_RESERVE}(
            0, // min_liquidity
            UNISWAP_INITIAL_TOKEN_RESERVE, // max_tokens
            DEADLINE // deadline
        );

        // Ensure Uniswap exchange is working as expected
        assertEq(
            uniswapExchange.getTokenToEthInputPrice(1 ether),
            calculateTokenToEthInputPrice(
                1 ether,
                UNISWAP_INITIAL_TOKEN_RESERVE,
                UNISWAP_INITIAL_ETH_RESERVE
            )
        );

        // Setup initial token balances of pool and attacker account
        dvt.transfer(attacker, ATTACKER_INITIAL_TOKEN_BALANCE);
        dvt.transfer(address(puppetPool), POOL_INITIAL_TOKEN_BALANCE);

        // Ensure correct setup of pool.
        assertEq(
            puppetPool.calculateDepositRequired(POOL_INITIAL_TOKEN_BALANCE),
            POOL_INITIAL_TOKEN_BALANCE * 2
        );
        vm.startPrank(attacker);
        dvt.approve(address(uniswapExchange), 2 ** 256 - 1);
        
        // console.log(unicode"🧨 PREPARED TO BREAK THINGS 🧨");
    }

    // A given helper function
    // Calculates how much ETH (in wei) Uniswap will pay for the given amount of tokens
    function calculateTokenToEthInputPrice(
        uint256 input_amount,
        uint256 input_reserve,
        uint256 output_reserve
    ) internal returns (uint256) {
        uint256 input_amount_with_fee = input_amount * 997;
        uint256 numerator = input_amount_with_fee * output_reserve;
        uint256 denominator = (input_reserve * 1000) + input_amount_with_fee;
        return numerator / denominator;
    }




    function testExample0() public {
        // Action: SwapUniswapETH2DVT
        uniswapExchange.ethToTokenSwapInput{value: 1 * 1e18}(1, 0xffffffff);
    }

    function testExample1() public {
        // Action: SwapUniswapDVT2ETH
        uniswapExchange.tokenToEthSwapInput(1 * 1e18, 1, 0xffffffff);
    }

    function testExample2() public {
        // Action: PoolBorrow
        ethGot = 1 * 1e18;
        ethNeed = puppetPool.calculateDepositRequired( ethGot );
        puppetPool.borrow{value: ethNeed}( ethGot );
    }






}
